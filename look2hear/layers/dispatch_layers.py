from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DISPATCHLoss(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        patch_size: int | None = None,
        top_k_percent: float = 0.3,
        low_freq_patch_size: int | None = None,
        high_freq_patch_size: int | None = None,
        split_freq_bin: int | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.patch_size = patch_size
        self.top_k_percent = top_k_percent
        self.low_freq_patch_size = low_freq_patch_size
        self.high_freq_patch_size = high_freq_patch_size
        self.split_freq_bin = split_freq_bin
        self.eps = eps

    def forward(
        self,
        student_waveform: torch.Tensor,
        teacher_waveform: torch.Tensor,
        target_waveform: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        student_mag = self._magnitude(student_waveform)
        teacher_mag = self._magnitude(teacher_waveform)
        target_mag = self._magnitude(target_waveform)

        patch_loss, kgs = self._compute_patch_loss_and_kgs(
            student_mag, teacher_mag, target_mag
        )

        # 只对 top-k patch 计算梯度（真正的 patch 级加权蒸馏）。
        bsz, num_patches = patch_loss.shape
        k = max(1, int(num_patches * float(self.top_k_percent)))
        _, idx = torch.topk(kgs, k=k, dim=1, largest=True)
        mask = torch.zeros_like(kgs).scatter_(1, idx, 1.0)
        selected = (patch_loss * mask).sum(1)
        count = mask.sum(1).clamp(min=1.0)
        loss = (selected / count).mean()

        flat_kgs = kgs.reshape(-1)
        stats = {
            "selected_patches_ratio": float(mask.mean().detach().cpu()),
            "kgs_mean": float(flat_kgs.mean().detach().cpu()),
            "kgs_pos_ratio": float((flat_kgs > 0).float().mean().detach().cpu()),
        }
        return loss, stats

    def _magnitude(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3:
            waveform = waveform.reshape(-1, waveform.shape[-1])
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
        )
        return spec.abs().clamp_min(self.eps)

    def _patchify(self, magnitude: torch.Tensor, patch_size: int) -> torch.Tensor:
        freq_bins, time_steps = magnitude.shape[-2], magnitude.shape[-1]
        pad_f = (patch_size - freq_bins % patch_size) % patch_size
        pad_t = (patch_size - time_steps % patch_size) % patch_size
        padded = F.pad(magnitude, (0, pad_t, 0, pad_f))
        unfolded = padded.unfold(-2, patch_size, patch_size).unfold(
            -1, patch_size, patch_size
        )
        return unfolded.contiguous().reshape(magnitude.shape[0], -1, patch_size * patch_size)

    def _compute_patch_loss_and_kgs(
        self,
        student_mag: torch.Tensor,
        teacher_mag: torch.Tensor,
        target_mag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 支持两种模式：
        # 1) 兼容旧配置：单尺度 patch_size（self.patch_size）
        # 2) MSSP：低频大 patch、高频小 patch（low/high/split）
        if (
            self.low_freq_patch_size is None
            or self.high_freq_patch_size is None
            or self.split_freq_bin is None
        ):
            if self.patch_size is None:
                raise ValueError("DISPATCHLoss requires patch_size or MSSP patch sizes.")
            ps = int(self.patch_size)
            s_p = self._patchify(student_mag, ps)
            t_p = self._patchify(teacher_mag, ps)
            tgt_p = self._patchify(target_mag, ps)
            err_s = (s_p - tgt_p).abs().mean(dim=-1)
            err_t = (t_p - tgt_p).abs().mean(dim=-1)
            kgs = err_s - err_t
            patch_loss = F.mse_loss(s_p, t_p, reduction="none").mean(dim=-1)
            return patch_loss, kgs

        split = int(self.split_freq_bin)
        low_ps = int(self.low_freq_patch_size)
        high_ps = int(self.high_freq_patch_size)

        s_low = student_mag[:, :split]
        t_low = teacher_mag[:, :split]
        tgt_low = target_mag[:, :split]
        s_high = student_mag[:, split:]
        t_high = teacher_mag[:, split:]
        tgt_high = target_mag[:, split:]

        s_pl = self._patchify(s_low, low_ps)
        t_pl = self._patchify(t_low, low_ps)
        tgt_pl = self._patchify(tgt_low, low_ps)
        s_ph = self._patchify(s_high, high_ps)
        t_ph = self._patchify(t_high, high_ps)
        tgt_ph = self._patchify(tgt_high, high_ps)

        err_s_l = (s_pl - tgt_pl).abs().mean(dim=-1)
        err_t_l = (t_pl - tgt_pl).abs().mean(dim=-1)
        err_s_h = (s_ph - tgt_ph).abs().mean(dim=-1)
        err_t_h = (t_ph - tgt_ph).abs().mean(dim=-1)
        kgs = torch.cat([err_s_l - err_t_l, err_s_h - err_t_h], dim=1)

        pl_l = F.mse_loss(s_pl, t_pl, reduction="none").mean(dim=-1)
        pl_h = F.mse_loss(s_ph, t_ph, reduction="none").mean(dim=-1)
        patch_loss = torch.cat([pl_l, pl_h], dim=1)
        return patch_loss, kgs
