import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_yaml(path: str):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def test_binary_config_exposes_design_stage_keys():
    config = _load_yaml("configs/tiger-small-binary.yml")
    assert "activation_warmup" in config["training"]["binary_stage_epochs"]
    assert "weight_binarize" in config["training"]["binary_stage_epochs"]
    binary_config = config["audionet"]["audionet_config"]["binary_config"]
    assert binary_config["protect_1x1_conv"] is False
    assert binary_config["protect_output_layer"] is True
    assert "bandsplit.proj" in binary_config["protect_patterns"]
    assert "q_proj" in binary_config["protect_patterns"]
    assert "mask_gen" in binary_config["protect_patterns"]


def test_kaggle_binary_config_exposes_design_stage_keys():
    config = _load_yaml("configs/tiger-small-kaggle-t4x2-binary.yml")
    assert "activation_warmup" in config["training"]["binary_stage_epochs"]
    assert "weight_binarize" in config["training"]["binary_stage_epochs"]
    binary_config = config["audionet"]["audionet_config"]["binary_config"]
    assert binary_config["protect_1x1_conv"] is False
    assert binary_config["protect_output_layer"] is True
