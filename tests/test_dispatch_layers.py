import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from look2hear.layers.dispatch_layers import DISPATCHLoss


def test_dispatch_loss_returns_scalar_and_stats():
    criterion = DISPATCHLoss(n_fft=320, hop_length=160, patch_size=4, top_k_percent=0.25)
    student = torch.randn(2, 3200)
    teacher = torch.randn(2, 3200)
    target = torch.randn(2, 3200)
    loss, stats = criterion(student, teacher, target)
    assert loss.ndim == 0
    assert "selected_patches_ratio" in stats
    assert "kgs_pos_ratio" in stats


def test_dispatch_selects_teacher_better_regions():
    # 当 teacher==target 时，teacher 误差为 0，student 随机，KGS 应该大多为正。
    criterion = DISPATCHLoss(n_fft=320, hop_length=160, patch_size=4, top_k_percent=0.3)
    target = torch.randn(2, 3200)
    teacher = target.clone()
    student = torch.randn(2, 3200)
    _, stats = criterion(student, teacher, target)
    assert stats["kgs_pos_ratio"] > 0.8
