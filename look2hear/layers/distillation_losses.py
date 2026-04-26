from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionLoss(nn.Module):
    """分布对齐损失：对齐 LayerNorm 后的特征分布形状而非绝对幅度。"""

    def __init__(self, temp: float = 1.0):
        super().__init__()
        self.temp = float(temp)

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        # 约定输入为 [B, C, T]（本仓库中通常是分离后 waveform: [B, S, T]，也可视作 C=S）。
        if student.shape != teacher.shape:
            raise ValueError(
                f"DistributionLoss expects same shapes, got student={tuple(student.shape)} "
                f"teacher={tuple(teacher.shape)}"
            )
        s = F.layer_norm(student, student.shape[1:]) / self.temp
        t = F.layer_norm(teacher, teacher.shape[1:]) / self.temp
        return F.mse_loss(s, t)

