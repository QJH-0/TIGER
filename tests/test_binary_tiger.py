import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from look2hear.models.binary_tiger import BinaryTIGER
from look2hear.layers.binary_layers import BinaryConv2d


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


def test_binary_tiger_converts_conv2d_layers():
    """验证 BinaryTIGER 转换 Conv2d 层（FFI Q/K/V 投影和 concat_block）。"""
    model = BinaryTIGER(
        sample_rate=16000,
        out_channels=16,
        in_channels=32,
        num_blocks=2,
        upsampling_depth=2,
        win=640,
        stride=160,
        num_sources=2,
        binary_config={"protect_attention": False, "protect_first_layer": False},
    )

    # 统计 BinaryConv2d 模块数量
    conv2d_count = sum(1 for m in model.modules() if isinstance(m, BinaryConv2d))
    assert conv2d_count > 0, "BinaryTIGER 应该转换至少一个 Conv2d 层"

    # 验证 concat_block 中的 Conv2d 被转换
    converted_names = model.converter.converted_modules
    has_concat_block = any("concat_block" in name for name in converted_names)
    assert has_concat_block, "concat_block 的 Conv2d 应被转换"
