"""选择性二值化测试。

验证 TIGERBinaryConverter 的 binarize_scope 参数能按模块类别选择性二值化。
"""

import pytest
import torch.nn as nn
from look2hear.utils.model_converter import TIGERBinaryConverter
from look2hear.layers.binary_layers import BinaryConv1d, BinaryConv2d


class DummyModel(nn.Module):
    """模拟 TIGER 模块路径结构的简单模型（含 Conv1d 和 Conv2d）。"""

    def __init__(self):
        super().__init__()
        # BandSplit 投影层
        self.bandsplit = nn.ModuleDict({
            "proj_0": nn.Conv1d(10, 20, 1),
            "proj_1": nn.Conv1d(10, 20, 1),
        })
        # mask 生成头
        self.mask = nn.ModuleDict({
            "head_0": nn.Conv1d(20, 10, 1),
        })
        # Depthwise 卷积
        self.spp_dw = nn.ModuleDict({
            "dw_0": nn.Conv1d(20, 20, 3, padding=1, groups=20),
        })
        # Pointwise 卷积
        self.proj_1x1 = nn.Conv1d(20, 20, 1)
        self.res_conv = nn.Conv1d(20, 20, 1)
        # F3A 注意力层（Conv2d，模拟 FFI 中的 Q/K/V 投影）
        self.queries = nn.Conv2d(20, 20, 1)
        self.keys = nn.Conv2d(20, 20, 1)
        # concat_block（深度可分离 Conv2d）
        self.concat_block = nn.Sequential(
            nn.Conv2d(20, 20, 1, 1, groups=20), nn.PReLU()
        )
        # 普通卷积
        self.normal_conv = nn.Conv1d(20, 20, 3, padding=1)


def _count_binary_modules(model: nn.Module) -> int:
    """统计 BinaryConv1d + BinaryConv2d 模块数量。"""
    return sum(1 for m in model.modules() if isinstance(m, (BinaryConv1d, BinaryConv2d)))


class TestClassifyModule:
    """测试 _classify_module 静态方法。"""

    def test_bn_classification(self):
        assert TIGERBinaryConverter._classify_module("bandsplit.proj_0") == "bn"
        assert TIGERBinaryConverter._classify_module("bandsplit.proj_1") == "bn"

    def test_mask_classification(self):
        assert TIGERBinaryConverter._classify_module("mask.head_0") == "mask"

    def test_dw_classification(self):
        assert TIGERBinaryConverter._classify_module("spp_dw.dw_0") == "dw"

    def test_pw_classification(self):
        assert TIGERBinaryConverter._classify_module("proj_1x1") == "pw"
        assert TIGERBinaryConverter._classify_module("res_conv") == "pw"

    def test_f3a_classification(self):
        assert TIGERBinaryConverter._classify_module("queries") == "f3a"
        assert TIGERBinaryConverter._classify_module("keys") == "f3a"

    def test_unknown_returns_none(self):
        assert TIGERBinaryConverter._classify_module("normal_conv") is None
        assert TIGERBinaryConverter._classify_module("some_other_layer") is None


class TestBinarizeScope:
    """测试 binarize_scope 选择性二值化。"""

    def test_empty_scope_converts_all(self):
        """空 scope 时应全量转换（默认保护策略）。"""
        model = DummyModel()
        converter = TIGERBinaryConverter(
            protect_first_layer=False,
            protect_attention=False,
            protect_output_layer=False,
            binarize_scope=[],
        )
        converter.convert(model)
        # 应该转换多个模块
        assert len(converter.converted_modules) > 0

    def test_scope_bn_only(self):
        """仅 bn 类别时，只有 BandSplit 相关层被转换。"""
        model = DummyModel()
        converter = TIGERBinaryConverter(
            protect_first_layer=False,
            protect_output_layer=False,
            binarize_scope=["bn"],
        )
        converter.convert(model)
        # 只有 bandsplit.proj 被转换
        for name in converter.converted_modules:
            assert "bandsplit" in name and "proj" in name

    def test_scope_mask_only(self):
        """仅 mask 类别时，只有 mask 相关层被转换。"""
        model = DummyModel()
        converter = TIGERBinaryConverter(
            protect_first_layer=False,
            protect_output_layer=False,
            binarize_scope=["mask"],
        )
        converter.convert(model)
        for name in converter.converted_modules:
            assert "mask" in name

    def test_scope_dw_only(self):
        """仅 dw 类别时，只有 Depthwise 卷积被转换。"""
        model = DummyModel()
        converter = TIGERBinaryConverter(
            protect_first_layer=False,
            protect_output_layer=False,
            binarize_scope=["dw"],
        )
        converter.convert(model)
        for name in converter.converted_modules:
            assert "spp_dw" in name or "dwconv" in name

    def test_scope_pw_only(self):
        """仅 pw 类别时，只有 Pointwise 卷积被转换。"""
        model = DummyModel()
        converter = TIGERBinaryConverter(
            protect_first_layer=False,
            protect_output_layer=False,
            binarize_scope=["pw"],
        )
        converter.convert(model)
        for name in converter.converted_modules:
            assert any(kw in name for kw in ["proj_1x1", "fc1", "fc2", "loc_glo_fus", "res_conv"])

    def test_scope_f3a_only(self):
        """仅 f3a 类别时，只有注意力层被转换。"""
        model = DummyModel()
        converter = TIGERBinaryConverter(
            protect_first_layer=False,
            protect_output_layer=False,
            binarize_scope=["f3a"],
        )
        converter.convert(model)
        for name in converter.converted_modules:
            assert any(kw in name for kw in ["queries", "keys", "values", "attn_concat_proj"])

    def test_scope_combined(self):
        """组合 scope 时，多个类别都被转换。"""
        model = DummyModel()
        converter = TIGERBinaryConverter(
            protect_first_layer=False,
            protect_output_layer=False,
            binarize_scope=["bn", "mask"],
        )
        converter.convert(model)
        # 应有 bandsplit 和 mask 相关模块
        has_bn = any("bandsplit" in name for name in converter.converted_modules)
        has_mask = any("mask" in name for name in converter.converted_modules)
        assert has_bn
        assert has_mask

    def test_scope_empty_falls_back_to_full(self):
        """空 scope 时行为与原版一致。"""
        model = DummyModel()
        converter_full = TIGERBinaryConverter(
            protect_first_layer=False,
            protect_attention=False,
            protect_output_layer=False,
            binarize_scope=[],
        )
        converter_full.convert(model)
        full_count = len(converter_full.converted_modules)

        # 空 scope 应该转换所有非保护模块
        assert full_count > 0

    def test_scope_with_protect_patterns(self):
        """scope 与 protect_patterns 交互：protect_patterns 优先。"""
        model = DummyModel()
        converter = TIGERBinaryConverter(
            protect_first_layer=False,
            protect_output_layer=False,
            binarize_scope=["bn"],
            protect_patterns=["bandsplit.proj_0"],
        )
        converter.convert(model)
        # bandsplit.proj_0 被 protect_patterns 保护，不被转换
        for name in converter.converted_modules:
            assert "proj_0" not in name
