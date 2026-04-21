import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from look2hear.layers.binary_layers import BinaryConv1d, RPReLU, RSign


def test_rsign_supports_bct():
    x = torch.randn(2, 8, 16)
    y = RSign(8)(x)
    assert y.shape == x.shape


def test_rsign_uses_learnable_alpha_threshold():
    layer = RSign(4)
    layer.alpha.data.fill_(0.25)
    x = torch.tensor([[[0.2], [0.3], [0.24], [0.26]]])
    out = layer(x)
    assert out.tolist() == [[[-1.0], [1.0], [-1.0], [1.0]]]


def test_rprelu_supports_bct():
    x = torch.randn(2, 8, 16)
    y = RPReLU(8)(x)
    assert y.shape == x.shape


def test_rprelu_exposes_beta_gamma_zeta_parameters():
    layer = RPReLU(3)
    assert tuple(layer.beta.shape) == (1, 3, 1)
    assert tuple(layer.gamma.shape) == (1, 3, 1)
    assert tuple(layer.zeta.shape) == (1, 3, 1)


def test_binary_conv1d_keeps_output_shape():
    x = torch.randn(2, 8, 16)
    layer = BinaryConv1d(8, 12, 3, padding=1)
    y = layer(x)
    assert y.shape == (2, 12, 16)
