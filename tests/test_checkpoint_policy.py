import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_test import resolve_eval_model_source
from audio_train import (
    build_checkpoint_callback,
    resolve_cli_resume_override,
    resolve_export_checkpoint_path,
    resolve_resume_checkpoint_path,
)


def test_build_checkpoint_callback_keeps_only_last_and_single_best():
    # 断言 checkpoint 策略符合当前约定：仅保留 last.ckpt 与单个 best.ckpt。
    checkpoint = build_checkpoint_callback("D:/tmp/exp")

    assert Path(checkpoint.dirpath) == Path("D:/tmp/exp")
    assert checkpoint.filename == "best"
    assert checkpoint.monitor == "val/loss"
    assert checkpoint.every_n_epochs == 1
    assert checkpoint.save_last is True
    assert checkpoint.save_top_k == 1


def test_resolve_resume_checkpoint_path_prefers_last_ckpt_when_resume_enabled(tmp_path):
    # resume_from_checkpoint=true 时，默认恢复当前实验目录下的 last.ckpt。
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    last_ckpt = exp_dir / "last.ckpt"
    last_ckpt.write_text("checkpoint")

    config = {"main_args": {"resume_from_checkpoint": True}}

    assert resolve_resume_checkpoint_path(config, str(exp_dir)) == str(last_ckpt)


def test_resolve_resume_checkpoint_path_uses_explicit_ckpt_path():
    # 显式给出 .ckpt 路径时，应直接使用该路径。
    config = {"main_args": {"resume_from_checkpoint": "D:/custom/model.ckpt"}}

    assert resolve_resume_checkpoint_path(config, "D:/tmp/exp") == "D:/custom/model.ckpt"


def test_resolve_cli_resume_override_uses_last_when_resume_flag_enabled():
    plain_args = types.SimpleNamespace(resume=True, resume_ckpt=None)

    assert resolve_cli_resume_override(plain_args) is True


def test_resolve_cli_resume_override_prefers_explicit_resume_ckpt():
    plain_args = types.SimpleNamespace(resume=True, resume_ckpt="D:/custom/from-cli.ckpt")

    assert resolve_cli_resume_override(plain_args) == "D:/custom/from-cli.ckpt"


def test_resolve_resume_checkpoint_path_uses_cli_merged_value_over_original_config(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    last_ckpt = exp_dir / "last.ckpt"
    last_ckpt.write_text("checkpoint")

    config = {
        "main_args": {
            "resume_from_checkpoint": True,
        }
    }

    assert resolve_resume_checkpoint_path(config, str(exp_dir)) == str(last_ckpt)


def test_resolve_eval_model_source_defaults_to_best_model(tmp_path):
    # 正常评估场景默认读取 best_model.pth，而不是训练 checkpoint。
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    source = resolve_eval_model_source(
        {"train_conf": {"exp": {"exp_name": "demo"}}},
        str(exp_dir),
    )

    assert source["source_type"] == "best"
    assert source["path"] == str(exp_dir / "best_model.pth")


def test_resolve_eval_model_source_supports_pretrained_override(tmp_path):
    # 显式指定 test_model_path 时，评估入口应切到预训练模型分支。
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    config = {
        "train_conf": {
            "exp": {"exp_name": "demo"},
            "main_args": {"test_model_path": "JusperLee/TIGER-speech"},
        }
    }

    source = resolve_eval_model_source(config, str(exp_dir))

    assert source == {"source_type": "pretrained", "path": "JusperLee/TIGER-speech"}


def test_resolve_export_checkpoint_path_prefers_best_model_path(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    best_ckpt = exp_dir / "epoch=9.ckpt"
    best_ckpt.write_text("checkpoint")
    checkpoint = types.SimpleNamespace(best_model_path=str(best_ckpt), last_model_path="")

    assert resolve_export_checkpoint_path(checkpoint, str(exp_dir)) == str(best_ckpt)


def test_resolve_export_checkpoint_path_falls_back_to_last_ckpt(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    last_ckpt = exp_dir / "last.ckpt"
    last_ckpt.write_text("checkpoint")
    checkpoint = types.SimpleNamespace(best_model_path="", last_model_path=str(last_ckpt))

    assert resolve_export_checkpoint_path(checkpoint, str(exp_dir)) == str(last_ckpt)
