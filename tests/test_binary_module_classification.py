import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from look2hear.layers.binary_layers import classify_binary_module


def test_classify_binary_module_recognizes_converted_bn_paths():
    assert classify_binary_module("BN.0.1") == "bn"
    assert classify_binary_module("model.BN.3.1.2") == "bn"
