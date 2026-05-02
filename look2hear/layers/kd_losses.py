"""D1/D2/D3 知识蒸馏损失（PIT 感知），对应《蒸馏技术方案》。

- D1: SI-SNR 输出蒸馏（黑盒，PIT 对齐后计算 SI-SNR）
- D2: 子带选择性蒸馏（PIT 对齐后，频率加权 MSE）
- D3: 输出 + 子带联合蒸馏（共享同一 PIT 排列）
"""

from __future__ import annotations

from itertools import permutations
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
#  工具函数
# ---------------------------------------------------------------------------

def _si_snr(ests: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算 SI-SNR（Scale-Invariant Signal-to-Noise Ratio）。

    Args:
        ests: [B, S, T] 估计信号
        targets: [B, S, T] 目标信号
    Returns:
        [B] 每个样本的平均 SI-SNR（线性，非 dB）
    """
    # 零均值
    ests = ests - ests.mean(dim=-1, keepdim=True)
    targets = targets - targets.mean(dim=-1, keepdim=True)
    # 按源计算 SI-SNR: s_target = <est, target> * target / ||target||^2
    dot = (ests * targets).sum(dim=-1, keepdim=True)  # [B, S, 1]
    target_energy = (targets ** 2).sum(dim=-1, keepdim=True) + eps  # [B, S, 1]
    s_target = dot * targets / target_energy  # [B, S, T]
    noise = ests - s_target
    si_snr = (s_target ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps)  # [B, S]
    return si_snr.mean(dim=-1)  # [B]


def pit_si_snr_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """PIT 感知的 SI-SNR 蒸馏损失。

    寻找使 SI-SNR 最大化的排列 pi*，返回 -SI-SNR(S_pi*, T)。

    Args:
        student: [B, K, T] 学生输出
        teacher: [B, K, T] 教师输出
    Returns:
        标量损失（负 SI-SNR 的均值）
    """
    B, K, T = student.shape
    if K == 1:
        return -_si_snr(student, teacher, eps).mean()

    # 枚举所有排列（K=2 时只有 2 种，K=3 时 6 种）
    perms = list(permutations(range(K)))
    best_si_snr = None
    for perm in perms:
        student_perm = student[:, list(perm), :]  # [B, K, T]
        si_snr = _si_snr(student_perm, teacher, eps)  # [B]
        if best_si_snr is None:
            best_si_snr = si_snr
        else:
            best_si_snr = torch.maximum(best_si_snr, si_snr)
    return -best_si_snr.mean()


def pit_align(
    student: torch.Tensor,
    teacher: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """PIT 对齐：返回按最优排列重排后的 student 输出。

    Args:
        student: [B, K, ...] 学生输出
        teacher: [B, K, ...] 教师输出（可为任意形状，只要 K 维一致）
    Returns:
        student_aligned: [B, K, ...] 按最优排列重排后的学生输出
    """
    B = student.shape[0]
    K = student.shape[1]
    if K == 1:
        return student

    # 用时域波形计算 SI-SNR 来确定排列
    T_dim = student.shape[-1]
    perms = list(permutations(range(K)))
    best_score = None
    best_perm_idx = 0
    for i, perm in enumerate(perms):
        student_perm = student[:, list(perm), :]
        si_snr = _si_snr(student_perm, teacher, eps)  # [B]
        if best_score is None:
            best_score = si_snr
            best_perm_idx = torch.full((B,), i, device=student.device)
        else:
            mask = si_snr > best_score
            best_perm_idx = torch.where(mask, torch.full_like(best_perm_idx, i), best_perm_idx)
            best_score = torch.maximum(best_score, si_snr)

    # 按每个样本的最优排列重排
    result = torch.empty_like(student)
    for b in range(B):
        perm = perms[best_perm_idx[b].item()]
        result[b] = student[b, list(perm), :]
    return result


# ---------------------------------------------------------------------------
#  子带权重工具
# ---------------------------------------------------------------------------

def build_subband_weights(
    band_width: Sequence[int],
    sample_rate: int = 16000,
    low_freq_weight: float = 2.0,
    mid_low_weight: float = 1.5,
    mid_weight: float = 1.0,
    mid_high_weight: float = 0.8,
    high_freq_weight: float = 0.5,
) -> torch.Tensor:
    """根据 TIGER 的 67 子带划分构建频率加权向量。

    子带划分（win=640, stride=160, sr=16000）：
    - 索引 0-39: 25Hz 窄带（低频），权重 2.0
    - 索引 40-49: 100Hz 带（中低频），权重 1.5
    - 索引 50-57: 250Hz 带（中频），权重 1.0
    - 索引 58-65: 500Hz 带（中高频），权重 0.8
    - 索引 66: 剩余（高频），权重 0.5

    Returns:
        [nband] 权重向量，不做归一化
    """
    nband = len(band_width)
    weights = torch.zeros(nband)
    for i in range(nband):
        if i < 40:
            weights[i] = low_freq_weight
        elif i < 50:
            weights[i] = mid_low_weight
        elif i < 58:
            weights[i] = mid_weight
        elif i < 66:
            weights[i] = mid_high_weight
        else:
            weights[i] = high_freq_weight
    return weights


def _stft_subband_mse(
    ests: torch.Tensor,
    targets: torch.Tensor,
    band_width: Sequence[int],
    n_fft: int,
    hop_length: int,
    weights: torch.Tensor,
    window: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算子带级加权 MSE：STFT -> 按子带分组 -> 加权 MSE。

    Args:
        ests: [B*K, T] 波形
        targets: [B*K, T] 波形
        band_width: 67 子带的频点宽度列表
        n_fft: STFT 窗口大小
        hop_length: STFT hop
        weights: [nband] 子带权重
        window: 预计算的窗函数（可选，避免重复创建）
    Returns:
        标量损失
    """
    if window is None:
        window = torch.hann_window(n_fft, device=ests.device, dtype=ests.dtype)
    est_spec = torch.stft(ests, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    tgt_spec = torch.stft(targets, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)

    # [B*K, F, T] -> 复数 MSE 逐频点
    mse_per_bin = (est_spec - tgt_spec).abs().square()  # [B*K, F, T]

    # 按子带分组加权（取均值而非求和，避免损失量级随子带数膨胀）
    nband = len(band_width)
    loss = torch.zeros((), device=ests.device, dtype=ests.dtype)
    freq_idx = 0
    for i in range(nband):
        bw = band_width[i]
        band_mse = mse_per_bin[:, freq_idx:freq_idx + bw, :].mean()  # 标量
        loss = loss + weights[i] * band_mse
        freq_idx += bw
    return loss / nband


# ---------------------------------------------------------------------------
#  D1: SI-SNR 输出蒸馏（PIT 感知）
# ---------------------------------------------------------------------------

class SI_SNR_KDLoss(nn.Module):
    """D1: SI-SNR 输出蒸馏损失（PIT 感知）。

    蒸馏对象为最终分离波形，通过 PIT 对齐后计算 SI-SNR。
    L_kd = -SI-SNR(S_pi*, T)，其中 pi* 为最优排列。
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student: [B, K, T] 学生分离波形
            teacher: [B, K, T] 教师分离波形
        Returns:
            标量蒸馏损失
        """
        return pit_si_snr_loss(student, teacher)


# ---------------------------------------------------------------------------
#  D2: 子带选择性蒸馏（PIT 感知）
# ---------------------------------------------------------------------------

class Subband_KDLoss(nn.Module):
    """D2: 子带选择性蒸馏损失（PIT 感知）。

    PIT 对齐后，对 TIGER 的 67 子带施加不同蒸馏权重（低频高、高频低）。
    L_band = sum_i w_i * MSE(S_pi*(i), T(i))
    权重不做归一化，直接加权以真实放大低频贡献。
    """

    def __init__(
        self,
        band_width: Sequence[int],
        n_fft: int = 640,
        hop_length: int = 160,
        low_freq_weight: float = 2.0,
        mid_low_weight: float = 1.5,
        mid_weight: float = 1.0,
        mid_high_weight: float = 0.8,
        high_freq_weight: float = 0.5,
    ):
        super().__init__()
        self.band_width = list(band_width)
        self.n_fft = n_fft
        self.hop_length = hop_length
        weights = build_subband_weights(
            band_width,
            low_freq_weight=low_freq_weight,
            mid_low_weight=mid_low_weight,
            mid_weight=mid_weight,
            mid_high_weight=mid_high_weight,
            high_freq_weight=high_freq_weight,
        )
        # 注册为 buffer，跟随设备但不参与梯度
        self.register_buffer("band_weights", weights)
        self.register_buffer("stft_window", torch.hann_window(n_fft))

    def forward(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student: [B, K, T] 学生分离波形
            teacher: [B, K, T] 教师分离波形
        Returns:
            标量蒸馏损失
        """
        B, K, T = student.shape

        # Step 1: PIT 对齐
        student_aligned = pit_align(student, teacher)  # [B, K, T]

        # Step 2: 逐源计算子带加权 MSE
        total_loss = torch.zeros((), device=student.device, dtype=student.dtype)
        for s in range(K):
            loss_s = _stft_subband_mse(
                student_aligned[:, s, :],  # [B, T]
                teacher[:, s, :],           # [B, T]
                self.band_width,
                self.n_fft,
                self.hop_length,
                self.band_weights,
                window=self.stft_window,
            )
            total_loss = total_loss + loss_s
        return total_loss / K


# ---------------------------------------------------------------------------
#  D3: 输出 + 子带联合蒸馏（PIT 统一对齐）
# ---------------------------------------------------------------------------

class Combined_KDLoss(nn.Module):
    """D3: 输出 + 子带联合蒸馏损失（共享同一 PIT 排列）。

    联合 D1（全局 SI-SNR 输出蒸馏）与 D2（子带选择性蒸馏），
    但共享同一个通过 PIT 确定的最优排列，避免两个损失在不同排列下计算导致冲突。

    L_total = L_task + lambda_out * L_out + lambda_band * L_band
    """

    def __init__(
        self,
        band_width: Sequence[int],
        n_fft: int = 640,
        hop_length: int = 160,
        low_freq_weight: float = 2.0,
        mid_low_weight: float = 1.5,
        mid_weight: float = 1.0,
        mid_high_weight: float = 0.8,
        high_freq_weight: float = 0.5,
    ):
        super().__init__()
        self.band_width = list(band_width)
        self.n_fft = n_fft
        self.hop_length = hop_length
        weights = build_subband_weights(
            band_width,
            low_freq_weight=low_freq_weight,
            mid_low_weight=mid_low_weight,
            mid_weight=mid_weight,
            mid_high_weight=mid_high_weight,
            high_freq_weight=high_freq_weight,
        )
        self.register_buffer("band_weights", weights)
        self.register_buffer("stft_window", torch.hann_window(n_fft))

    def forward(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            student: [B, K, T] 学生分离波形
            teacher: [B, K, T] 教师分离波形
        Returns:
            (L_out, L_band) 两个标量损失，由调用方用 lambda 加权求和
        """
        B, K, T = student.shape

        # 统一 PIT 排列：用 SI-SNR 寻优
        student_aligned = pit_align(student, teacher)  # [B, K, T]

        # D1: 全局输出损失（负 SI-SNR）
        l_out = -_si_snr(student_aligned, teacher).mean()

        # D2: 子带损失
        l_band = torch.zeros((), device=student.device, dtype=student.dtype)
        for s in range(K):
            loss_s = _stft_subband_mse(
                student_aligned[:, s, :],
                teacher[:, s, :],
                self.band_width,
                self.n_fft,
                self.hop_length,
                self.band_weights,
                window=self.stft_window,
            )
            l_band = l_band + loss_s
        l_band = l_band / K

        return l_out, l_band
