import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_yaml(path: str):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def test_local_config_filenames_are_explicitly_prefixed():
    expected = [
        "configs/tiger-small-local.yml",
        "configs/tiger-small-local-binary.yml",
        "configs/tiger-small-local-binary-distill-1.yml",
        "configs/tiger-small-local-binary-distill-2.yml",
        "configs/tiger-small-local-binary-distill-3.yml",
        "configs/tiger-small-local-binary-S0.yml",
        "configs/tiger-small-local-binary-S1.yml",
        "configs/tiger-small-local-binary-S2.yml",
        "configs/tiger-small-local-binary-S3.yml",
        "configs/tiger-small-local-binary-S4.yml",
        "configs/tiger-small-local-binary-B1.yml",
        "configs/tiger-small-local-binary-B2.yml",
        "configs/tiger-small-local-binary-B3.yml",
        "configs/tiger-small-local-binary-B4.yml",
        "configs/tiger-small-local-binary-A1.yml",
        "configs/tiger-small-local-binary-A2.yml",
        "configs/tiger-small-local-binary-A3.yml",
        "configs/tiger-small-local-binary-A4.yml",
    ]
    for path in expected:
        assert Path(path).exists(), path


def test_kaggle_stage_config_filenames_are_explicitly_prefixed():
    expected = [
        "configs/tiger-small-kaggle-t4x2-binary-S0.yml",
        "configs/tiger-small-kaggle-t4x2-binary-S1.yml",
        "configs/tiger-small-kaggle-t4x2-binary-S2.yml",
        "configs/tiger-small-kaggle-t4x2-binary-S3.yml",
        "configs/tiger-small-kaggle-t4x2-binary-S4.yml",
        "configs/tiger-small-kaggle-t4x2-binary-B0.yml",
        "configs/tiger-small-kaggle-t4x2-binary-B1.yml",
        "configs/tiger-small-kaggle-t4x2-binary-B2.yml",
        "configs/tiger-small-kaggle-t4x2-binary-B3.yml",
        "configs/tiger-small-kaggle-t4x2-binary-B4.yml",
        "configs/tiger-small-kaggle-t4x2-binary-A1.yml",
        "configs/tiger-small-kaggle-t4x2-binary-A2.yml",
        "configs/tiger-small-kaggle-t4x2-binary-A3.yml",
        "configs/tiger-small-kaggle-t4x2-binary-A4.yml",
    ]
    for path in expected:
        assert Path(path).exists(), path


def test_binary_config_exposes_design_stage_keys():
    config = _load_yaml("configs/tiger-small-local-binary.yml")
    assert "activation_warmup" in config["training"]["binary_stage_epochs"]
    assert "weight_binarize" in config["training"]["binary_stage_epochs"]
    binary_config = config["audionet"]["audionet_config"]["binary_config"]
    assert binary_config["protect_1x1_conv"] is False
    assert binary_config["protect_output_layer"] is True
    assert "bandsplit.proj" in binary_config["protect_patterns"]
    assert "q_proj" in binary_config["protect_patterns"]
    assert "mask_gen" in binary_config["protect_patterns"]


def test_binary_distill_config_has_required_keys():
    """二值化+蒸馏配置包含所有必需键。"""
    config = _load_yaml("configs/tiger-small-kaggle-t4x2-binary-distill-3.yml")
    assert config["training"]["system"] == "BinaryDistillAudioLitModule"
    assert config["distillation"]["enabled"] is True
    assert config["distillation"]["kd_type"] in {"d1", "d2", "d3"}
    assert "teacher_ckpt" in config["distillation"]
    assert "student_init_ckpt" in config["distillation"]
    assert "lambda_out_range" in config["distillation"]
    assert "lambda_band_range" in config["distillation"]
    # 优化器应为 AdamW
    assert config["optimizer"]["optim_name"] == "adamw"
    assert config["optimizer"]["weight_decay"] == 1.0e-4
    # 参数分组
    assert "param_groups" in config["optimizer"]
    assert "fp32_lr" in config["optimizer"]["param_groups"]
    assert "binary_lr" in config["optimizer"]["param_groups"]
    # 调度器应为余弦退火
    assert config["scheduler"]["sche_name"] == "CosineAnnealingLR"


def test_kaggle_binary_config_exposes_design_stage_keys():
    config = _load_yaml("configs/tiger-small-kaggle-t4x2-binary-B0.yml")
    assert "activation_warmup" in config["training"]["binary_stage_epochs"]
    assert "weight_binarize" in config["training"]["binary_stage_epochs"]
    binary_config = config["audionet"]["audionet_config"]["binary_config"]
    assert binary_config["protect_1x1_conv"] is False
    assert binary_config["protect_output_layer"] is True


def test_a4_config_uses_subband_distillation_ablation_instead_of_switching_to_d1():
    """A4 应保留子带蒸馏，只关闭选择性权重。"""
    config = _load_yaml("configs/tiger-small-kaggle-t4x2-binary-A4.yml")
    distill = config["distillation"]
    assert distill["enabled"] is True
    assert distill["kd_type"] in {"d2", "d3"}
    assert "low_freq_weight" in distill
    assert "mid_low_weight" in distill
    assert "mid_weight" in distill
    assert "mid_high_weight" in distill
    assert "high_freq_weight" in distill
    assert distill["low_freq_weight"] == 1.0
    assert distill["mid_low_weight"] == 1.0
    assert distill["mid_weight"] == 1.0
    assert distill["mid_high_weight"] == 1.0
    assert distill["high_freq_weight"] == 1.0


def test_local_stage0_configs_depend_on_local_fp32_checkpoint_instead_of_kaggle_outputs():
    expected_fp32 = "Experiments/TIGER-MiniLibriMix-Local/best_model.pth"
    for name in ("S0", "S1", "S2", "S3", "S4"):
        config = _load_yaml(f"configs/tiger-small-local-binary-{name}.yml")
        assert config["training"]["warmup_ckpt"] == expected_fp32


def test_local_distill_configs_default_to_local_checkpoints():
    expected_teacher = "Experiments/TIGER-MiniLibriMix-Local/best_model.pth"
    expected_student = "Experiments/TIGER-Small-Binary-Local/best_model.pth"
    for idx in (1, 2, 3):
        config = _load_yaml(f"configs/tiger-small-local-binary-distill-{idx}.yml")
        distill = config["distillation"]
        assert distill["teacher_ckpt"] == expected_teacher
        assert distill["student_init_ckpt"] == expected_student
        assert distill["distill_warmup_epochs"] == 5
