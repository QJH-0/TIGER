import torch
import torch.nn as nn
import torch.nn.functional as F


def classify_binary_module(full_name: str) -> str | None:
    """根据模块路径名返回模块类别（共享分类逻辑）。

    类别定义：
    - "bn": BandSplit 投影层（bandsplit.proj，不含 recover）
    - "mask": mask 生成头
    - "dw": Depthwise 卷积（spp_dw, dwconv, last_layer 中的 DW Conv）
    - "pw": Pointwise 卷积（proj_1x1, fc1/fc2, loc_glo_fus, res_conv）
    - "f3a": F3A 注意力层（queries, keys, values, attn_concat_proj）
    """
    name_lower = full_name.lower()

    # F3A 注意力层
    f3a_keywords = {"queries", "keys", "values", "attn_concat_proj"}
    segments = set(name_lower.split("."))
    if segments.intersection(f3a_keywords):
        return "f3a"

    # BandSplit 投影层（排除 recover）
    # 转换后的 TIGER 中，这组模块会挂在顶层 `BN.*` 路径下；
    # stage0 的 freeze_scope=["bn"] 依赖这里正确识别这些真实模块名。
    if "bn" in segments:
        return "bn"
    if "bandsplit" in name_lower or "band_split" in name_lower:
        if "proj" in name_lower and "recover" not in name_lower:
            return "bn"

    # mask 生成头
    if "mask" in name_lower:
        return "mask"

    # Depthwise 卷积
    dw_keywords = {"spp_dw", "dwconv", "dw_conv"}
    if any(kw in name_lower for kw in dw_keywords):
        return "dw"
    # last_layer 中的 DW Conv（groups > 1）
    if "last_layer" in name_lower and "conv" in name_lower:
        return "dw"

    # Pointwise 卷积
    pw_keywords = {"proj_1x1", "fc1", "fc2", "loc_glo_fus", "res_conv"}
    if any(kw in name_lower for kw in pw_keywords):
        return "pw"

    return None


def _clipped_ste_sign(x: torch.Tensor) -> torch.Tensor:
    """Clipped STE（Bi-RealNet）：前向始终输出 {-1, +1}，反向梯度仅在 |w| <= 1 时传递。"""
    w_bin = torch.sign(x)
    # 前向：直接返回 sign 结果
    # 反向：仅在 |x| <= 1 时传递梯度（mask=1），其余 mask=0
    mask = (x.abs() <= 1.0).float()
    return w_bin.detach() + x * mask - (x * mask).detach()


class RSign(nn.Module):
    """可学习阈值的激活二值化模块。

    alpha 形状为 [1, channels, 1]，前向时自动扩展维度以兼容 3D/4D 输入。
    """
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))
        self.disabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            return x
        # 将 alpha 扩展到与 x 相同的维度数，支持 3D [B,C,T] 和 4D [B,C,H,W]
        alpha = self.alpha
        while alpha.ndim < x.ndim:
            alpha = alpha.unsqueeze(-1)
        signed = torch.where(x >= alpha, torch.ones_like(x), -torch.ones_like(x))
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
    """`nn.Conv1d` 的二值版本（ReActNet 框架）。

    关键设计：
    - 权重用 Clipped STE 做符号投影（Bi-RealNet）；
    - Scale factor 用 EMA 更新（衰减 0.9），训练时按 optimizer.step() 节奏同步；
    - Sign + Scale 计算强制 FP32，避免 16-mixed 精度损失。
    """

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
        ema_decay: float = 0.9,
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
        self.ema_decay = float(ema_decay)
        if self.use_scale:
            self.register_buffer("weight_scale", torch.ones(out_channels, 1, 1))
            self._scale_initialized = False

    def init_scale_from_weights(self) -> None:
        """用当前权重的 l1 均值初始化 Scale（调用一次，在加载预训练权重后）。"""
        if not self.use_scale:
            return
        with torch.no_grad():
            scale = self.weight.float().abs().reshape(self.weight.size(0), -1).mean(1, keepdim=True).unsqueeze(-1)
            self.weight_scale.copy_(scale)
            self._scale_initialized = True

    def update_ema_scale(self) -> None:
        """EMA 更新 Scale：alpha_t = decay * alpha_{t-1} + (1 - decay) * ||W||_1。
        每次 optimizer.step() 后调用一次。
        """
        if not self.use_scale or not self.training:
            return
        with torch.no_grad():
            w = self.weight.float()
            instant_scale = w.abs().reshape(w.size(0), -1).mean(1, keepdim=True).unsqueeze(-1)
            if not self._scale_initialized:
                self.weight_scale.copy_(instant_scale)
                self._scale_initialized = True
            else:
                self.weight_scale.mul_(self.ema_decay).add_(instant_scale, alpha=1.0 - self.ema_decay)

    def _binary_weight(self) -> torch.Tensor:
        """前向：Clipped STE 符号投影 + Scale，Sign/Scale 强制 FP32。"""
        if not self.use_scale:
            with torch.amp.autocast("cuda", enabled=False):
                w_fp32 = self.weight.float()
                w_ste = _clipped_ste_sign(w_fp32)
            return w_ste

        # Sign + Scale 计算强制 FP32
        with torch.amp.autocast("cuda", enabled=False):
            w_fp32 = self.weight.float()
            w_ste = _clipped_ste_sign(w_fp32)
            scale = self.weight_scale.float()
            result = w_ste * scale
        return result

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


class BinaryConv2d(nn.Conv2d):
    """`nn.Conv2d` 的二值版本（ReActNet 框架）。

    关键设计与 BinaryConv1d 一致：
    - 权重用 Clipped STE 做符号投影（Bi-RealNet）；
    - Scale factor 用 EMA 更新（衰减 0.9），训练时按 optimizer.step() 节奏同步；
    - Sign + Scale 计算强制 FP32，避免 16-mixed 精度损失。

    用途：TIGER 模型中 FFI（MultiHeadSelfAttention2D）的 Q/K/V 投影和
    Recurrent.concat_block 的深度可分离 Conv2d 均为 nn.Conv2d，
    需要 BinaryConv2d 才能覆盖这些层的二值化。
    """

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
        ema_decay: float = 0.9,
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
        self.ema_decay = float(ema_decay)
        if self.use_scale:
            self.register_buffer("weight_scale", torch.ones(out_channels, 1, 1, 1))
            self._scale_initialized = False

    def init_scale_from_weights(self) -> None:
        """用当前权重的 l1 均值初始化 Scale（调用一次，在加载预训练权重后）。"""
        if not self.use_scale:
            return
        with torch.no_grad():
            scale = self.weight.float().abs().reshape(self.weight.size(0), -1).mean(1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            self.weight_scale.copy_(scale)
            self._scale_initialized = True

    def update_ema_scale(self) -> None:
        """EMA 更新 Scale：alpha_t = decay * alpha_{t-1} + (1 - decay) * ||W||_1。
        每次 optimizer.step() 后调用一次。
        """
        if not self.use_scale or not self.training:
            return
        with torch.no_grad():
            w = self.weight.float()
            instant_scale = w.abs().reshape(w.size(0), -1).mean(1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            if not self._scale_initialized:
                self.weight_scale.copy_(instant_scale)
                self._scale_initialized = True
            else:
                self.weight_scale.mul_(self.ema_decay).add_(instant_scale, alpha=1.0 - self.ema_decay)

    def _binary_weight(self) -> torch.Tensor:
        """前向：Clipped STE 符号投影 + Scale，Sign/Scale 强制 FP32。"""
        if not self.use_scale:
            with torch.amp.autocast("cuda", enabled=False):
                w_fp32 = self.weight.float()
                w_ste = _clipped_ste_sign(w_fp32)
            return w_ste

        # Sign + Scale 计算强制 FP32
        with torch.amp.autocast("cuda", enabled=False):
            w_fp32 = self.weight.float()
            w_ste = _clipped_ste_sign(w_fp32)
            scale = self.weight_scale.float()
            result = w_ste * scale
        return result

    def clamp_weights(self) -> None:
        with torch.no_grad():
            self.weight.clamp_(-1.0, 1.0)
            if self.bias is not None:
                self.bias.clamp_(-1.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self._binary_weight() if self.use_binary else self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class BinaryLinear(nn.Linear):
    """`nn.Linear` 的二值版本，支持 Clipped STE 和 EMA Scale。"""

    def __init__(self, *args, use_scale: bool = True, ema_decay: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_binary = True
        self.use_scale = bool(use_scale)
        self.ema_decay = float(ema_decay)
        if self.use_scale:
            self.register_buffer("weight_scale", torch.ones(self.out_features, 1))
            self._scale_initialized = False

    def init_scale_from_weights(self) -> None:
        if not self.use_scale:
            return
        with torch.no_grad():
            scale = self.weight.float().abs().reshape(self.weight.size(0), -1).mean(1, keepdim=True)
            self.weight_scale.copy_(scale)
            self._scale_initialized = True

    def update_ema_scale(self) -> None:
        if not self.use_scale or not self.training:
            return
        with torch.no_grad():
            w = self.weight.float()
            instant_scale = w.abs().reshape(w.size(0), -1).mean(1, keepdim=True)
            if not self._scale_initialized:
                self.weight_scale.copy_(instant_scale)
                self._scale_initialized = True
            else:
                self.weight_scale.mul_(self.ema_decay).add_(instant_scale, alpha=1.0 - self.ema_decay)

    def _binary_weight(self) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            w_fp32 = self.weight.float()
            w_ste = _clipped_ste_sign(w_fp32)
            if self.use_scale:
                scale = self.weight_scale.float()
                return w_ste * scale
            return w_ste

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
