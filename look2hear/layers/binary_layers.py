import torch
import torch.nn as nn
import torch.nn.functional as F


def _ste_sign(x: torch.Tensor) -> torch.Tensor:
    # 直通估计器：前向投影到 {-1, +1}，反向仍保留实值梯度。
    signed = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    return x + (signed - x).detach()


class RSign(nn.Module):
    # 可学习阈值的激活二值化模块。
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signed = torch.where(x >= self.alpha, torch.ones_like(x), -torch.ones_like(x))
        return x + (signed - x).detach()


class RPReLU(nn.Module):
    # 分布重塑模块：分别缩放正负半轴并加可学习偏移。
    def __init__(self, channels: int):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(1, channels, 1) * 0.5)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1) * 0.1)
        self.zeta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.maximum(x, torch.zeros_like(x))
        neg = torch.minimum(x, torch.zeros_like(x))
        return self.beta * pos + self.gamma * neg + self.zeta


class BinaryConv1d(nn.Conv1d):
    # `nn.Conv1d` 的二值版本，是 BinaryTIGER 的主要替换目标。
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_binary = True

    def _binary_weight(self) -> torch.Tensor:
        # 前向使用符号投影权重，优化器仍更新潜在实值权重。
        return _ste_sign(self.weight)

    def clamp_weights(self) -> None:
        with torch.no_grad():
            self.weight.clamp_(-1.0, 1.0)
            if self.bias is not None:
                self.bias.clamp_(-1.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(
            x,
            self._binary_weight() if self.use_binary else self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class BinaryLinear(nn.Linear):
    # `nn.Linear` 的二值版本，当前仓库中默认可选。
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_binary = True

    def _binary_weight(self) -> torch.Tensor:
        return _ste_sign(self.weight)

    def clamp_weights(self) -> None:
        with torch.no_grad():
            self.weight.clamp_(-1.0, 1.0)
            if self.bias is not None:
                self.bias.clamp_(-1.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self._binary_weight() if self.use_binary else self.weight,
            self.bias,
        )


class BinaryBlock(nn.Module):
    # 独立二值块：RSign + BinaryConv1d + RPReLU。
    # 当前 BinaryTIGER 路径主要直接替换 Conv1d/Linear，而不是重写为该块。
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.sign = RSign(in_channels)
        self.conv = BinaryConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.act = RPReLU(out_channels)

    def clamp_weights(self) -> None:
        self.conv.clamp_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(self.sign(x)))
