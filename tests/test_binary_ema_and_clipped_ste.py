"""BinaryConv1d EMA Scale 和 Clipped STE 测试。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from look2hear.layers.binary_layers import (
    BinaryConv1d,
    BinaryConv2d,
    BinaryLinear,
    _clipped_ste_sign,
)


def test_clipped_ste_sign_output_range():
    """Clipped STE 前向输出在 {-1, 0, +1}（sign(0)=0 是 PyTorch 行为）。"""
    x = torch.randn(2, 8, 16)
    y = _clipped_ste_sign(x)
    # sign(x) 输出 {-1, 0, +1}，其中 0 仅在 x 恰好为 0 时出现
    allowed = torch.tensor([-1.0, 0.0, 1.0])
    for val in y.flatten():
        assert any(torch.isclose(val, a, atol=1e-6) for a in allowed), f"Unexpected value: {val}"


def test_clipped_ste_sign_gradient_masked_outside_one():
    """Clipped STE：|w| > 1 的梯度应被 mask。"""
    x = torch.tensor([0.5, 1.5, -0.3, -2.0], requires_grad=True)
    y = _clipped_ste_sign(x)
    y.sum().backward()
    # x=[0.5] 在 [-1,1] 内，梯度保留；x=[1.5] 超出，梯度为 0
    assert x.grad[0].item() != 0.0  # 0.5 在范围内
    assert x.grad[1].item() == 0.0  # 1.5 超出范围
    assert x.grad[2].item() != 0.0  # -0.3 在范围内
    assert x.grad[3].item() == 0.0  # -2.0 超出范围


def test_binary_conv1d_ema_scale_init():
    """EMA Scale 初始化应从权重 l1 均值得到。"""
    layer = BinaryConv1d(8, 12, 3, padding=1, use_scale=True, ema_decay=0.9)
    # 手动设置权重
    layer.weight.data.fill_(0.5)
    layer.init_scale_from_weights()
    # Scale 应为 |0.5| = 0.5
    expected = 0.5
    assert torch.allclose(layer.weight_scale, torch.full_like(layer.weight_scale, expected), atol=1e-6)


def test_binary_conv1d_ema_scale_update():
    """EMA Scale 更新应平滑移动。"""
    layer = BinaryConv1d(8, 12, 3, padding=1, use_scale=True, ema_decay=0.9)
    layer.train()
    # 初始化
    layer.weight.data.fill_(1.0)
    layer.init_scale_from_weights()
    old_scale = layer.weight_scale.clone()
    # 改变权重
    layer.weight.data.fill_(2.0)
    layer.update_ema_scale()
    new_scale = layer.weight_scale.clone()
    # EMA: 0.9 * 1.0 + 0.1 * 2.0 = 1.1
    expected = 0.9 * 1.0 + 0.1 * 2.0
    assert torch.allclose(new_scale, torch.full_like(new_scale, expected), atol=1e-5)


def test_binary_conv1d_ema_no_update_in_eval():
    """eval 模式下 EMA Scale 不更新。"""
    layer = BinaryConv1d(8, 12, 3, padding=1, use_scale=True, ema_decay=0.9)
    layer.train()
    layer.weight.data.fill_(1.0)
    layer.init_scale_from_weights()
    layer.eval()
    layer.weight.data.fill_(5.0)
    layer.update_ema_scale()
    # eval 模式下不应更新
    assert torch.allclose(layer.weight_scale, torch.ones_like(layer.weight_scale), atol=1e-6)


def test_binary_conv1d_fp32_forward():
    """BinaryConv1d 前向在 FP32 下应正常工作。"""
    layer = BinaryConv1d(8, 12, 3, padding=1, use_scale=True)
    layer.init_scale_from_weights()
    x = torch.randn(2, 8, 16)
    y = layer(x)
    assert y.shape == (2, 12, 16)
    assert y.dtype == torch.float32


def test_binary_linear_scale():
    """BinaryLinear 支持 Scale Factor。"""
    layer = BinaryLinear(8, 12, use_scale=True, ema_decay=0.9)
    layer.weight.data.fill_(0.5)
    layer.init_scale_from_weights()
    expected = 0.5
    assert torch.allclose(layer.weight_scale, torch.full_like(layer.weight_scale, expected), atol=1e-6)


def test_binary_linear_ema_update():
    """BinaryLinear EMA Scale 更新。"""
    layer = BinaryLinear(8, 12, use_scale=True, ema_decay=0.9)
    layer.train()
    layer.weight.data.fill_(1.0)
    layer.init_scale_from_weights()
    layer.weight.data.fill_(2.0)
    layer.update_ema_scale()
    expected = 0.9 * 1.0 + 0.1 * 2.0
    assert torch.allclose(layer.weight_scale, torch.full_like(layer.weight_scale, expected), atol=1e-5)


def test_binary_conv1d_use_binary_toggle():
    """use_binary 开关控制是否使用二值权重。"""
    layer = BinaryConv1d(8, 12, 3, padding=1, use_scale=True)
    layer.init_scale_from_weights()
    x = torch.randn(2, 8, 16)

    layer.use_binary = True
    y_bin = layer(x)

    layer.use_binary = False
    y_fp = layer(x)

    # 二值和全精度输出应不同
    assert not torch.allclose(y_bin, y_fp, atol=1e-3)


def test_binary_conv2d_ema_scale_init():
    """BinaryConv2d EMA Scale 初始化应从权重 l1 均值得到。"""
    layer = BinaryConv2d(8, 12, 3, padding=1, use_scale=True, ema_decay=0.9)
    layer.weight.data.fill_(0.5)
    layer.init_scale_from_weights()
    expected = 0.5
    assert torch.allclose(layer.weight_scale, torch.full_like(layer.weight_scale, expected), atol=1e-6)


def test_binary_conv2d_ema_scale_update():
    """BinaryConv2d EMA Scale 更新应平滑移动。"""
    layer = BinaryConv2d(8, 12, 3, padding=1, use_scale=True, ema_decay=0.9)
    layer.train()
    layer.weight.data.fill_(1.0)
    layer.init_scale_from_weights()
    old_scale = layer.weight_scale.clone()
    layer.weight.data.fill_(2.0)
    layer.update_ema_scale()
    new_scale = layer.weight_scale.clone()
    expected = 0.9 * 1.0 + 0.1 * 2.0
    assert torch.allclose(new_scale, torch.full_like(new_scale, expected), atol=1e-5)


def test_binary_conv2d_fp32_forward():
    """BinaryConv2d 前向在 FP32 下应正常工作。"""
    layer = BinaryConv2d(8, 12, 3, padding=1, use_scale=True)
    layer.init_scale_from_weights()
    x = torch.randn(2, 8, 16, 16)
    y = layer(x)
    assert y.shape == (2, 12, 16, 16)
    assert y.dtype == torch.float32


def test_binary_conv2d_use_binary_toggle():
    """BinaryConv2d use_binary 开关控制是否使用二值权重。"""
    layer = BinaryConv2d(8, 12, 3, padding=1, use_scale=True)
    layer.init_scale_from_weights()
    x = torch.randn(2, 8, 16, 16)

    layer.use_binary = True
    y_bin = layer(x)

    layer.use_binary = False
    y_fp = layer(x)

    assert not torch.allclose(y_bin, y_fp, atol=1e-3)
