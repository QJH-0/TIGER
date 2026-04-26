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
        # 保守初始化：等价于恒等映射，避免训练初期破坏幅度分布。
        self.beta = nn.Parameter(torch.ones(1, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.zeta = nn.Parameter(torch.zeros(1, channels, 1))
        # 训练阶段开关：用于“先禁用、后启用”的渐进策略。
        self.active = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return x

        # 以 gamma 为阈值分割正负区间，并允许偏移 zeta。
        pos_mask = (x > self.gamma).float()
        neg_mask = 1.0 - pos_mask
        pos = (x - self.gamma + self.zeta) * pos_mask
        neg = (self.beta * (x - self.gamma) + self.zeta) * neg_mask
        return pos + neg


class BinaryConv1d(nn.Conv1d):
    # `nn.Conv1d` 的二值版本，是 BinaryTIGER 的主要替换目标。
    #
    # 关键点：
    # - 权重用 STE 做符号投影；
    # - scale factor 训练时更新、推理时冻结复用，避免漂移。
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
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        use_scale: bool = True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.use_binary = True
        self.use_scale = bool(use_scale)
        if self.use_scale:
            self.register_buffer("weight_scale", torch.ones(out_channels, 1, 1))

    def _binary_weight(self) -> torch.Tensor:
        # 前向使用符号投影权重，优化器仍更新潜在实值权重。
        w = self.weight
        w_bin = torch.sign(w)
        w_ste = w_bin + (w - w_bin).detach()
        if not self.use_scale:
            return w_ste

        if self.training:
            scale = w.abs().reshape(w.size(0), -1).mean(1, keepdim=True).unsqueeze(-1)
            self.weight_scale.copy_(scale.detach())
        else:
            scale = self.weight_scale
        return w_ste * scale

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


class ReActNetBlock(nn.Module):
    # ReActNet 基础块：BinaryConv1d + RSign + RPReLU，并支持残差捷径。
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
        self.conv = BinaryConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            use_scale=True,
        )
        self.rsign = RSign(out_channels)
        self.rprelu = RPReLU(out_channels)
        self.shortcut = (in_channels == out_channels and stride == 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv(x)
        out = self.rsign(out)
        out = self.rprelu(out)
        if self.shortcut:
            out = out + identity
        return out
