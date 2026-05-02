"""D1/D2/D3 蒸馏损失测试。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from look2hear.layers.kd_losses import (
    SI_SNR_KDLoss,
    Subband_KDLoss,
    Combined_KDLoss,
    pit_si_snr_loss,
    pit_align,
    build_subband_weights,
)


def test_pit_si_snr_loss_identity():
    """PIT SI-SNR 损失：相同信号损失应为负值（SI-SNR 为正）。"""
    B, K, T = 2, 2, 16000
    signal = torch.randn(B, K, T)
    loss = pit_si_snr_loss(signal, signal)
    # 相同信号 SI-SNR 应很高（损失很负）
    assert loss.item() < -10.0


def test_pit_si_snr_loss_permutation_invariance():
    """PIT 应对说话人排列不变。"""
    B, K, T = 2, 2, 16000
    teacher = torch.randn(B, K, T)
    # 学生是教师的排列
    student = teacher[:, [1, 0], :]
    loss = pit_si_snr_loss(student, teacher)
    # 排列后应完全匹配
    assert loss.item() < -10.0


def test_pit_si_snr_loss_noisy():
    """加噪声后损失应上升（SI-SNR 下降）。"""
    B, K, T = 2, 2, 16000
    teacher = torch.randn(B, K, T)
    student = teacher + 0.5 * torch.randn(B, K, T)
    loss_noisy = pit_si_snr_loss(student, teacher)
    loss_clean = pit_si_snr_loss(teacher, teacher)
    assert loss_noisy.item() > loss_clean.item()


def test_pit_align_preserves_shape():
    """PIT 对齐后形状不变。"""
    B, K, T = 2, 2, 16000
    student = torch.randn(B, K, T)
    teacher = torch.randn(B, K, T)
    aligned = pit_align(student, teacher)
    assert aligned.shape == student.shape


def test_pit_align_single_source():
    """单源时 PIT 对齐是恒等。"""
    B, K, T = 2, 1, 16000
    student = torch.randn(B, K, T)
    teacher = torch.randn(B, K, T)
    aligned = pit_align(student, teacher)
    assert torch.allclose(aligned, student)


def test_si_snr_kd_loss_output():
    """D1 损失输出标量。"""
    B, K, T = 2, 2, 4000
    student = torch.randn(B, K, T)
    teacher = torch.randn(B, K, T)
    loss_fn = SI_SNR_KDLoss()
    loss = loss_fn(student, teacher)
    assert loss.ndim == 0
    assert loss.item() != 0.0


def test_subband_kd_loss_output():
    """D2 损失输出标量且为正。"""
    band_width = [1] * 40 + [4] * 10 + [10] * 8 + [20] * 8 + [61]
    B, K, T = 2, 2, 16000
    student = torch.randn(B, K, T)
    teacher = torch.randn(B, K, T)
    loss_fn = Subband_KDLoss(band_width=band_width, n_fft=640, hop_length=160)
    loss = loss_fn(student, teacher)
    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_subband_kd_loss_zero_when_identical():
    """D2 损失：相同时应接近零。"""
    band_width = [1] * 40 + [4] * 10 + [10] * 8 + [20] * 8 + [61]
    B, K, T = 2, 2, 16000
    signal = torch.randn(B, K, T)
    loss_fn = Subband_KDLoss(band_width=band_width, n_fft=640, hop_length=160)
    loss = loss_fn(signal, signal)
    assert loss.item() < 1e-4


def test_combined_kd_loss_returns_two_scalars():
    """D3 损失返回 (L_out, L_band) 两个标量。"""
    band_width = [1] * 40 + [4] * 10 + [10] * 8 + [20] * 8 + [61]
    B, K, T = 2, 2, 4000
    student = torch.randn(B, K, T)
    teacher = torch.randn(B, K, T)
    loss_fn = Combined_KDLoss(band_width=band_width, n_fft=640, hop_length=160)
    l_out, l_band = loss_fn(student, teacher)
    assert l_out.ndim == 0
    assert l_band.ndim == 0
    assert l_out.item() != 0.0
    assert l_band.item() > 0.0


def test_build_subband_weights_shape():
    """子带权重向量形状正确。"""
    band_width = [1] * 40 + [4] * 10 + [10] * 8 + [20] * 8 + [61]
    weights = build_subband_weights(band_width)
    assert weights.shape == (67,)
    # 低频权重应高于高频
    assert weights[0] > weights[66]


def test_build_subband_weights_values():
    """子带权重按频段正确分配。"""
    band_width = [1] * 40 + [4] * 10 + [10] * 8 + [20] * 8 + [61]
    weights = build_subband_weights(band_width)
    assert abs(weights[0].item() - 2.0) < 1e-5   # 0-1kHz
    assert abs(weights[40].item() - 1.5) < 1e-5  # 1-2.5kHz
    assert abs(weights[50].item() - 1.0) < 1e-5  # 2.5-4.5kHz
    assert abs(weights[58].item() - 0.8) < 1e-5  # 4.5-8kHz
    assert abs(weights[66].item() - 0.5) < 1e-5  # 8kHz+


def test_subband_kd_loss_supports_uniform_weights():
    """A4: D2 应支持统一子带权重配置。"""
    band_width = [1] * 40 + [4] * 10 + [10] * 8 + [20] * 8 + [61]
    loss_fn = Subband_KDLoss(
        band_width=band_width,
        n_fft=640,
        hop_length=160,
        low_freq_weight=1.0,
        mid_low_weight=1.0,
        mid_weight=1.0,
        mid_high_weight=1.0,
        high_freq_weight=1.0,
    )
    assert torch.allclose(loss_fn.band_weights, torch.ones(67))


def test_combined_kd_loss_supports_uniform_weights():
    """A4: D3 也应支持统一子带权重配置。"""
    band_width = [1] * 40 + [4] * 10 + [10] * 8 + [20] * 8 + [61]
    loss_fn = Combined_KDLoss(
        band_width=band_width,
        n_fft=640,
        hop_length=160,
        low_freq_weight=1.0,
        mid_low_weight=1.0,
        mid_weight=1.0,
        mid_high_weight=1.0,
        high_freq_weight=1.0,
    )
    assert torch.allclose(loss_fn.band_weights, torch.ones(67))
