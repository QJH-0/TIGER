import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from look2hear.models.binary_tiger import BinaryTIGER


def test_binary_tiger_forward_runs():
    model = BinaryTIGER(
        sample_rate=16000,
        out_channels=16,
        in_channels=32,
        num_blocks=2,
        upsampling_depth=2,
        win=640,
        stride=160,
        num_sources=2,
        binary_config={"protect_attention": True},
    )
    x = torch.randn(1, 6400)
    y = model(x)
    assert y.ndim == 3


def test_binary_tiger_reports_binary_parameter_stats():
    model = BinaryTIGER(
        sample_rate=16000,
        out_channels=16,
        in_channels=32,
        num_blocks=2,
        upsampling_depth=2,
        win=640,
        stride=160,
        num_sources=2,
        binary_config={"protect_attention": True},
    )

    stats = model.get_binarization_summary()

    assert stats["binary_params"] > 0
    assert stats["total_params"] >= stats["binary_params"]
    assert stats["binary_ratio"] > 0
    assert len(stats["binary_module_names"]) > 0
