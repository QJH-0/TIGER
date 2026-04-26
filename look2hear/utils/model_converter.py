from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

from look2hear.layers.binary_layers import BinaryConv1d, BinaryLinear


@dataclass
class TIGERBinaryConverter:
    # 将原始 TIGER 转换为二值版本，同时保护脆弱结构。
    protect_first_layer: bool = True
    protect_attention: bool = True
    protect_1x1_conv: bool = True
    protect_output_layer: bool = True
    enable_binary_linear: bool = False
    protect_patterns: list[str] = field(default_factory=list)
    protected_modules: list[str] = field(default_factory=list)
    converted_modules: list[str] = field(default_factory=list)

    def convert(self, model: nn.Module) -> nn.Module:
        # 递归遍历模型并替换符合条件的模块。
        self.protected_modules = []
        self.converted_modules = []
        conv_state = {"seen_conv": False}
        self._convert_children(model, parent_name="", conv_state=conv_state)
        return model

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

        if self.enable_binary_linear and isinstance(module, nn.Linear):
            if self._is_attention_module(full_name, module):
                self.protected_modules.append(full_name)
                return module
            self.converted_modules.append(full_name)
            return self._to_binary_linear(module)

        return module

    def _should_protect_conv(
        self, module: nn.Conv1d, full_name: str, conv_state: dict[str, bool]
    ) -> bool:
        # 默认保护策略：
        # - 第一层特征投影
        # - 显式保护名称模式
        # - 注意力相关模块
        # - 1x1 投影
        # - mask / output 头
        if self.protect_first_layer and not conv_state["seen_conv"]:
            return True
        if self._matches_protect_pattern(full_name):
            return True
        if self.protect_attention and self._is_attention_module(full_name, module):
            return True
        if self.protect_1x1_conv and module.kernel_size == (1,):
            # 仅保护信息瓶颈/重构路径上的 1x1 投影，避免过度保护导致二值化覆盖率下降。
            full_name_l = full_name.lower()
            if "band" in full_name_l or "recover" in full_name_l:
                return True
        if self.protect_output_layer and self._is_output_module(full_name):
            return True
        return False

    def _matches_protect_pattern(self, full_name: str) -> bool:
        tokens = full_name.lower()
        return any(pattern.lower() in tokens for pattern in self.protect_patterns)

    @staticmethod
    def _is_attention_module(full_name: str, module: nn.Module) -> bool:
        # 注意：这里必须“严格判定”，不要把泛化子串（例如 "att"）都当成注意力层，
        # 否则像 "globalatt" 这类仅仅命名中带 att 的 MLP/全局分支也会被误判为注意力层，
        # 导致被保护、二值化覆盖率下降。
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
    def _to_binary_conv(module: nn.Conv1d) -> BinaryConv1d:
        # 仅替换结构，参数值直接复制自原始全精度层。
        converted = BinaryConv1d(
            module.in_channels,
            module.out_channels,
            module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
        )
        # BinaryConv1d 可能带额外 buffer（如 weight_scale）；从原始 Conv1d 加载时允许缺失。
        converted.load_state_dict(module.state_dict(), strict=False)
        return converted

    @staticmethod
    def _to_binary_linear(module: nn.Linear) -> BinaryLinear:
        # 仅替换结构，参数值直接复制自原始全精度层。
        converted = BinaryLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
        )
        converted.load_state_dict(module.state_dict())
        return converted
