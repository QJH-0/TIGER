import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from look2hear.models.tiger import TIGER
from look2hear.utils.model_converter import TIGERBinaryConverter


def test_converter_can_convert_tiger_model():
    model = TIGER(
        out_channels=16,
        in_channels=32,
        num_blocks=2,
        upsampling_depth=2,
        win=320,
        stride=160,
        num_sources=2,
        sample_rate=16000,
    )
    converted = TIGERBinaryConverter().convert(model)
    assert converted is not None


def test_converter_supports_named_protect_patterns():
    converter = TIGERBinaryConverter(protect_patterns=["bandsplit.proj", "mask"])
    assert converter._matches_protect_pattern("bandsplit.proj.0.conv")
    assert converter._matches_protect_pattern("separator.mask_gen")


def test_converter_protects_qkv_modules_even_when_1x1_is_disabled():
    converter = TIGERBinaryConverter(
        protect_attention=True,
        protect_1x1_conv=False,
        protect_patterns=["q_proj", "k_proj", "v_proj"],
    )
    module = nn.Conv1d(8, 8, 1)
    assert converter._should_protect_conv(module, "separator.block.q_proj", {"seen_conv": True})
