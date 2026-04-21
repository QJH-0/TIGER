from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DISPATCHLoss(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        patch_size: int,
        top_k_percent: float,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.patch_size = patch_size
        self.top_k_percent = top_k_percent
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

        student_patches = self._patchify(student_mag)
        teacher_patches = self._patchify(teacher_mag)
        target_patches = self._patchify(target_mag)

        # Knowledge Gap Score (KGS):
        # 只蒸馏 teacher 明显优于 student 的区域。
        # 定义为：gap = err_student - err_teacher；gap 越大表示 teacher 越值得蒸馏。
        err_student = (student_patches - target_patches).abs().mean(dim=-1)
        err_teacher = (teacher_patches - target_patches).abs().mean(dim=-1)
        kgs = err_student - err_teacher
        flat_kgs = kgs.reshape(-1)
        num_patches = flat_kgs.numel()
        top_k = max(1, int(num_patches * self.top_k_percent))
        # 仅选择 teacher 优于 student 的 patch；如果没有，则退化为选 top-k（避免空集合导致训练中断）。
        positive = torch.nonzero(flat_kgs > 0, as_tuple=False).reshape(-1)
        if positive.numel() > 0:
            k = min(top_k, int(positive.numel()))
            selected = positive[torch.topk(flat_kgs[positive], k=k, largest=True).indices]
        else:
            selected = torch.topk(flat_kgs, k=top_k, largest=True).indices

        student_flat = student_patches.reshape(num_patches, -1)
        teacher_flat = teacher_patches.reshape(num_patches, -1)
        loss = F.l1_loss(student_flat[selected], teacher_flat[selected])

        stats = {
            "selected_patches_ratio": float(selected.numel() / num_patches),
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

    def _patchify(self, magnitude: torch.Tensor) -> torch.Tensor:
        freq_bins, time_steps = magnitude.shape[-2], magnitude.shape[-1]
        pad_f = (self.patch_size - freq_bins % self.patch_size) % self.patch_size
        pad_t = (self.patch_size - time_steps % self.patch_size) % self.patch_size
        padded = F.pad(magnitude, (0, pad_t, 0, pad_f))
        unfolded = padded.unfold(-2, self.patch_size, self.patch_size).unfold(
            -1, self.patch_size, self.patch_size
        )
        return unfolded.contiguous().reshape(magnitude.shape[0], -1, self.patch_size * self.patch_size)
