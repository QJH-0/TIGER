from __future__ import annotations

from copy import deepcopy
from .base_model import BaseModel
from .tiger import TIGER
from ..layers.binary_layers import BinaryConv1d, BinaryLinear
from ..utils.model_converter import TIGERBinaryConverter


class BinaryTIGER(BaseModel):
    # Binary wrapper around the original TIGER.
    # Paper mapping:
    # - the architecture itself is still the original TIGER
    # - binarization is an implementation transform applied after construction
    # - selected Conv1d/Linear carriers inside the original paper blocks are
    #   converted, while protected structures stay full precision
    def __init__(self, sample_rate=44100, binary_config=None, **tiger_kwargs):
        super().__init__(sample_rate=sample_rate)
        self.binary_config = dict(binary_config or {})
        self.model_kwargs = dict(tiger_kwargs)
        self.model_kwargs["sample_rate"] = sample_rate
        # Build the original paper model first.
        self.model = TIGER(**self.model_kwargs)
        # Then convert eligible layers to binary counterparts.
        self.converter = TIGERBinaryConverter(**self.binary_config)
        self.model = self.converter.convert(self.model)

    def forward(self, wav, mouth=None):
        return self.model(wav)

    def clamp_all_binary_weights(self) -> None:
        # Keep latent weights inside a binary-friendly range between optimizer
        # steps.
        for module in self.modules():
            if isinstance(module, (BinaryConv1d, BinaryLinear)):
                module.clamp_weights()

    def set_binary_training(self, enabled: bool) -> None:
        # Warmup runs full-precision weights; binary stage switches forward to
        # sign-projected weights.
        for module in self.modules():
            if isinstance(module, (BinaryConv1d, BinaryLinear)):
                module.use_binary = enabled

    def get_binarization_summary(self):
        # 统计二值化覆盖（按参数个数）并给出多口径大小估计：
        # - fp32_estimated_*：按 FP32(4 bytes/param) 的粗略估计（与 Lightning Summary 口径一致）
        # - actual_*：按当前参数 dtype 的真实字节数（训练时可能是 fp16/bf16/fp32 混合）
        # - packed_binary_*：假设二值层“权重 1-bit 打包”，bias 仍 FP32，其余层 FP32 的推理端等效大小
        binary_params = 0
        total_params = sum(parameter.numel() for parameter in self.parameters())

        binary_weight_params = 0
        binary_bias_params = 0
        for module in self.modules():
            if isinstance(module, (BinaryConv1d, BinaryLinear)):
                # 这里的 "binary_params" 仍按模块参数总数统计（含 bias），用于覆盖率展示。
                binary_params += sum(parameter.numel() for parameter in module.parameters())
                # 但“打包估计”只把 weight 当成 1-bit，bias 仍按 FP32。
                if hasattr(module, "weight") and module.weight is not None:
                    binary_weight_params += int(module.weight.numel())
                if hasattr(module, "bias") and module.bias is not None:
                    binary_bias_params += int(module.bias.numel())

        ratio = float(binary_params / total_params) if total_params else 0.0

        # 1) Lightning 风格 FP32 粗估：总参数 * 4 bytes
        fp32_estimated_bytes = int(total_params) * 4

        # 2) 真实参数占用：按 dtype element_size 统计
        actual_bytes = int(
            sum(int(p.numel()) * int(p.element_size()) for p in self.parameters())
        )

        # 3) 二值打包等效估计：
        # - 非二值部分：仍按 FP32（简化口径，便于和论文/表格统一）
        # - 二值部分：weight 1-bit 打包（向上按字节对齐），bias 按 FP32
        nonbinary_params = int(total_params) - int(binary_params)
        packed_binary_weight_bytes = (int(binary_weight_params) + 7) // 8
        packed_binary_bias_bytes = int(binary_bias_params) * 4
        packed_binary_bytes = nonbinary_params * 4 + packed_binary_weight_bytes + packed_binary_bias_bytes

        def _mb(num_bytes: int) -> float:
            return round(float(num_bytes) / (1024.0 * 1024.0), 4)

        return {
            "binary_module_names": deepcopy(self.converter.converted_modules),
            "protected_module_names": deepcopy(self.converter.protected_modules),
            "binary_params": int(binary_params),
            "binary_weight_params": int(binary_weight_params),
            "binary_bias_params": int(binary_bias_params),
            "total_params": int(total_params),
            "binary_ratio": ratio,
            "binary_ratio_percent": round(ratio * 100.0, 2),
            "fp32_estimated_bytes": fp32_estimated_bytes,
            "fp32_estimated_size_mb": _mb(fp32_estimated_bytes),
            "actual_param_bytes": actual_bytes,
            "actual_param_size_mb": _mb(actual_bytes),
            "packed_binary_bytes": int(packed_binary_bytes),
            "packed_binary_size_mb": _mb(int(packed_binary_bytes)),
        }

    def get_model_args(self):
        return deepcopy(
            {
                **self.model_kwargs,
                "binary_config": deepcopy(self.binary_config),
            }
        )
