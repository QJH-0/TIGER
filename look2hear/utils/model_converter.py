"""TIGER 二值化转换器。

将原始 TIGER 模型转换为二值版本，支持：
- 全量转换（默认保护策略）
- 选择性转换（按模块类别 binarize_scope）
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import torch.nn as nn

from look2hear.layers.binary_layers import BinaryConv1d, BinaryConv2d, BinaryLinear, RSign, RPReLU, classify_binary_module


@dataclass
class TIGERBinaryConverter:
    """将原始 TIGER 转换为二值版本，同时保护脆弱结构。

    参数：
    - protect_first_layer: 保护第一层特征投影
    - protect_attention: 保护注意力相关模块
    - protect_1x1_conv: 保护信息瓶颈/重构路径上的 1x1 投影
    - protect_output_layer: 保护 mask/output 头
    - enable_binary_linear: 是否二值化 Linear 层
    - protect_patterns: 显式保护的名称模式列表
    - binarize_scope: 选择性二值化的模块类别列表，为空时全量转换
      支持的类别: "bn", "mask", "dw", "pw", "f3a"
    """
    protect_first_layer: bool = True
    protect_attention: bool = True
    protect_1x1_conv: bool = True
    protect_output_layer: bool = True
    enable_binary_linear: bool = False
    protect_patterns: list[str] = field(default_factory=list)
    binarize_scope: list[str] = field(default_factory=list)
    protected_modules: list[str] = field(default_factory=list)
    converted_modules: list[str] = field(default_factory=list)
    rsign_wrapped_modules: list[tuple[str, str]] = field(default_factory=list)

    def convert(self, model: nn.Module) -> nn.Module:
        """递归遍历模型并替换符合条件的模块。

        步骤：
        1. Conv1d/Conv2d/Linear → BinaryConv1d/BinaryConv2d/BinaryLinear
        2. 在 Sequential 容器中的 BinaryConv 前插入 RSign
        3. 将 nn.PReLU 替换为 RPReLU
        """
        self.protected_modules = []
        self.converted_modules = []
        self.rsign_wrapped_modules = []
        conv_state = {"seen_conv": False}
        self._convert_children(model, parent_name="", conv_state=conv_state)
        self._insert_rsign_before_binary_conv(model, parent_name="")
        self._replace_prelu_with_rprelu(model, parent_name="")
        return model

    def _insert_rsign_before_binary_conv(
        self, module: nn.Module, parent_name: str
    ) -> None:
        """在 Sequential 容器中的 BinaryConv 前插入 RSign。"""
        for child_name, child in list(module.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Sequential):
                self._insert_rsign_in_sequential(child, full_name)
            else:
                self._insert_rsign_before_binary_conv(child, full_name)

    def _insert_rsign_in_sequential(
        self, seq: nn.Sequential, seq_name: str
    ) -> None:
        """在 Sequential 内部的 BinaryConv 前插入 RSign。"""
        items = list(seq._modules.items())
        new_modules = []
        i = 0
        while i < len(items):
            name, mod = items[i]
            full_name = f"{seq_name}.{name}"
            if isinstance(mod, (BinaryConv1d, BinaryConv2d)):
                # 检查前一个模块是否已经是 RSign
                already_has_rsign = (
                    len(new_modules) > 0
                    and isinstance(new_modules[-1][1], RSign)
                )
                if not already_has_rsign:
                    rsign = RSign(mod.in_channels)
                    rsign_name = f"rsign_{name}"
                    new_modules.append((rsign_name, rsign))
                    self.rsign_wrapped_modules.append(
                        (full_name, f"{seq_name}.{rsign_name}")
                    )
            # 递归处理嵌套的 Sequential
            if isinstance(mod, nn.Sequential):
                self._insert_rsign_in_sequential(mod, full_name)
            new_modules.append((name, mod))
            i += 1
        # 重建 Sequential 的 _modules
        seq._modules = OrderedDict(new_modules)

    def _replace_prelu_with_rprelu(
        self, module: nn.Module, parent_name: str
    ) -> None:
        """递归将 nn.PReLU 替换为 RPReLU。"""
        for child_name, child in list(module.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.PReLU):
                channels = child.num_parameters
                rprelu = RPReLU(channels)
                setattr(module, child_name, rprelu)
            else:
                self._replace_prelu_with_rprelu(child, full_name)

    def _convert_children(
        self, module: nn.Module, parent_name: str, conv_state: dict[str, bool]
    ) -> None:
        for child_name, child in list(module.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            replacement = self._maybe_convert_module(child, full_name, conv_state)
            if replacement is not child:
                setattr(module, child_name, replacement)
                child = replacement
            self._convert_children(child, full_name, conv_state)

    def _maybe_convert_module(
        self, module: nn.Module, full_name: str, conv_state: dict[str, bool]
    ) -> nn.Module:
        if isinstance(module, nn.Conv1d):
            if self._should_protect_conv(module, full_name, conv_state):
                self.protected_modules.append(full_name)
                conv_state["seen_conv"] = True
                return module
            conv_state["seen_conv"] = True
            self.converted_modules.append(full_name)
            return self._to_binary_conv(module)

        if isinstance(module, nn.Conv2d):
            if self._should_protect_conv(module, full_name, conv_state):
                self.protected_modules.append(full_name)
                conv_state["seen_conv"] = True
                return module
            conv_state["seen_conv"] = True
            self.converted_modules.append(full_name)
            return self._to_binary_conv2d(module)

        if self.enable_binary_linear and isinstance(module, nn.Linear):
            if self._is_attention_module(full_name, module):
                self.protected_modules.append(full_name)
                return module
            self.converted_modules.append(full_name)
            return self._to_binary_linear(module)

        return module

    def _should_protect_conv(
        self, module: nn.Module, full_name: str, conv_state: dict[str, bool]
    ) -> bool:
        """判断是否应保护该卷积层。

        如果 binarize_scope 非空，则只有匹配 scope 类别的模块才会被转换，其余全部保护。
        如果 binarize_scope 为空，则使用默认保护策略。
        """
        # 选择性二值化模式：只有匹配 scope 的模块才会被转换
        if self.binarize_scope:
            category = self._classify_module(full_name)
            if category not in self.binarize_scope:
                return True  # 不在 scope 内，保护
            # 在 scope 内，继续检查是否属于硬性保护层
            if self._is_hard_protected(full_name, module, conv_state):
                return True
            return False

        # 全量转换模式：使用默认保护策略
        if self.protect_first_layer and not conv_state["seen_conv"]:
            return True
        if self._matches_protect_pattern(full_name):
            return True
        if self.protect_attention and self._is_attention_module(full_name, module):
            return True
        if self.protect_1x1_conv and module.kernel_size == (1,):
            full_name_l = full_name.lower()
            if "band" in full_name_l or "recover" in full_name_l:
                return True
        if self.protect_output_layer and self._is_output_module(full_name):
            return True
        return False

    def _is_hard_protected(
        self, full_name: str, module: nn.Module, conv_state: dict[str, bool]
    ) -> bool:
        """检查是否属于硬性保护层（始终不二值化）。

        - 第一层特征投影
        - mask/output 头
        - 显式保护名称模式
        """
        if self.protect_first_layer and not conv_state["seen_conv"]:
            return True
        if self._matches_protect_pattern(full_name):
            return True
        if self.protect_output_layer and self._is_output_module(full_name):
            return True
        return False

    @staticmethod
    def _classify_module(full_name: str) -> str | None:
        """根据模块路径名返回模块类别（委托给共享函数）。"""
        return classify_binary_module(full_name)

    def _matches_protect_pattern(self, full_name: str) -> bool:
        tokens = full_name.lower()
        return any(pattern.lower() in tokens for pattern in self.protect_patterns)

    @staticmethod
    def _is_attention_module(full_name: str, module: nn.Module) -> bool:
        full_name_l = full_name.lower()
        if "globalatt" in full_name_l:
            return False

        segments = set(full_name_l.split("."))
        attention_segments = {
            "attention",
            "attn",
            "qkv",
            "q_proj",
            "k_proj",
            "v_proj",
            "queries",
            "keys",
            "values",
            "attn_concat_proj",
        }
        if segments.intersection(attention_segments):
            return True

        class_name = module.__class__.__name__.lower()
        return "attention" in class_name

    @staticmethod
    def _is_output_module(full_name: str) -> bool:
        tokens = full_name.lower()
        return "mask" in tokens or tokens.endswith("output") or ".output" in tokens

    @staticmethod
    def _resolve_padding(padding):
        """将 nn.Conv 的 padding 属性转为构造器可接受的值。

        nn.Conv 的 padding 属性在 "same"/"valid" 模式下仍为字符串，
        需要直接传递；在数值模式下为元组，取第一个元素。
        """
        if isinstance(padding, str):
            return padding
        return padding[0]

    @staticmethod
    def _to_binary_conv(module: nn.Conv1d) -> BinaryConv1d:
        converted = BinaryConv1d(
            module.in_channels,
            module.out_channels,
            module.kernel_size[0],
            stride=module.stride[0],
            padding=TIGERBinaryConverter._resolve_padding(module.padding),
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
        )
        converted.load_state_dict(module.state_dict(), strict=False)
        return converted

    @staticmethod
    def _to_binary_conv2d(module: nn.Conv2d) -> BinaryConv2d:
        converted = BinaryConv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size[0],
            stride=module.stride[0],
            padding=TIGERBinaryConverter._resolve_padding(module.padding),
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
        )
        converted.load_state_dict(module.state_dict(), strict=False)
        return converted

    @staticmethod
    def _to_binary_linear(module: nn.Linear) -> BinaryLinear:
        converted = BinaryLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
        )
        converted.load_state_dict(module.state_dict())
        return converted
