import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_test import resolve_eval_model_source
from audio_train import (
    build_checkpoint_callback,
    build_resume_summary_message,
    cleanup_training_runtime,
    enforce_checkpoint_load_policy,
    extract_model_state_dict,
    extract_resume_progress,
    normalize_checkpoint_keys_for_model,
    resolve_datamodule_runtime_config,
    resolve_trainer_runtime_config,
    resolve_cli_resume_override,
    resolve_export_checkpoint_path,
    resolve_resume_checkpoint_path,
    summarize_checkpoint_load,
)


def test_build_checkpoint_callback_keeps_only_last_and_single_best():
    # 断言 checkpoint 策略符合当前约定：仅保留 last.ckpt 与单个 best.ckpt。
    checkpoint = build_checkpoint_callback("D:/tmp/exp/checkpoints")

    assert Path(checkpoint.dirpath) == Path("D:/tmp/exp/checkpoints")
    assert checkpoint.filename == "best"
    assert checkpoint.monitor == "val/loss"
    assert checkpoint.every_n_epochs == 1
    assert checkpoint.save_last is True
    assert checkpoint.save_top_k == 1
    # Lightning 将 enable_version_counter 存为私有字段。
    assert checkpoint._enable_version_counter is False


def test_resolve_resume_checkpoint_path_prefers_last_ckpt_when_resume_enabled(tmp_path):
    # resume_from_checkpoint=true 时，默认恢复 checkpoints/last.ckpt。
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    ck_dir = exp_dir / "checkpoints"
    ck_dir.mkdir()
    last_ckpt = ck_dir / "last.ckpt"
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
    ck_dir = exp_dir / "checkpoints"
    ck_dir.mkdir()
    last_ckpt = ck_dir / "last.ckpt"
    last_ckpt.write_text("checkpoint")

    config = {
        "main_args": {
            "resume_from_checkpoint": True,
        }
    }

    assert resolve_resume_checkpoint_path(config, str(exp_dir)) == str(last_ckpt)


def test_resolve_resume_checkpoint_path_falls_back_to_legacy_exp_root_last_ckpt(tmp_path):
    """旧版将 last.ckpt 放在实验根目录时仍可续训。"""
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    last_ckpt = exp_dir / "last.ckpt"
    last_ckpt.write_text("checkpoint")

    config = {"main_args": {"resume_from_checkpoint": "last"}}

    assert resolve_resume_checkpoint_path(config, str(exp_dir)) == str(last_ckpt)


def test_resolve_resume_prefers_checkpoints_subdir_when_both_last_ckpt_exist(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "last.ckpt").write_text("legacy")
    ck_dir = exp_dir / "checkpoints"
    ck_dir.mkdir()
    preferred = ck_dir / "last.ckpt"
    preferred.write_text("current")

    config = {"main_args": {"resume_from_checkpoint": True}}

    assert resolve_resume_checkpoint_path(config, str(exp_dir)) == str(preferred)


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
    ck_dir = exp_dir / "checkpoints"
    ck_dir.mkdir()
    last_ckpt = ck_dir / "last.ckpt"
    last_ckpt.write_text("checkpoint")
    checkpoint = types.SimpleNamespace(best_model_path="", last_model_path=str(last_ckpt))

    assert resolve_export_checkpoint_path(checkpoint, str(exp_dir)) == str(last_ckpt)


def test_resolve_export_checkpoint_path_falls_back_to_legacy_root_last_ckpt(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    last_ckpt = exp_dir / "last.ckpt"
    last_ckpt.write_text("checkpoint")
    checkpoint = types.SimpleNamespace(best_model_path="", last_model_path="")

    assert resolve_export_checkpoint_path(checkpoint, str(exp_dir)) == str(last_ckpt)


def test_extract_model_state_dict_supports_serialized_best_model_payload():
    payload = {
        "model_name": "BinaryTIGER",
        "state_dict": {
            "model.weight": 1,
            "model.bias": 2,
        },
        "model_args": {"sample_rate": 16000},
    }

    state_dict = extract_model_state_dict(payload)

    assert state_dict == {
        "model.weight": 1,
        "model.bias": 2,
    }


def test_summarize_checkpoint_load_counts_matched_missing_and_unexpected_keys():
    model = types.SimpleNamespace(
        state_dict=lambda: {
            "layer.weight": object(),
            "layer.bias": object(),
        }
    )
    incoming_state_dict = {
        "layer.weight": object(),
        "unused.weight": object(),
    }
    missing = ["layer.bias"]
    unexpected = ["unused.weight"]

    summary = summarize_checkpoint_load(model, incoming_state_dict, missing, unexpected)

    assert summary["model_key_count"] == 2
    assert summary["checkpoint_key_count"] == 2
    assert summary["matched_key_count"] == 1
    assert summary["missing_key_count"] == 1
    assert summary["unexpected_key_count"] == 1
    assert summary["matched_key_ratio"] == 0.5


def test_normalize_checkpoint_keys_for_binary_tiger_adds_model_prefix():
    model = types.SimpleNamespace(
        model=object(),
        state_dict=lambda: {
            "model.BN.0.0.weight": object(),
            "model.separator.mask.weight": object(),
        },
    )
    state_dict = {
        "BN.0.0.weight": 1,
        "separator.mask.weight": 2,
    }

    normalized = normalize_checkpoint_keys_for_model(model, state_dict)

    assert normalized == {
        "model.BN.0.0.weight": 1,
        "model.separator.mask.weight": 2,
    }


def test_normalize_checkpoint_keys_for_binary_tiger_keeps_prefixed_keys():
    model = types.SimpleNamespace(
        model=object(),
        state_dict=lambda: {
            "model.BN.0.0.weight": object(),
            "model.separator.mask.weight": object(),
        },
    )
    state_dict = {
        "model.BN.0.0.weight": 1,
        "model.separator.mask.weight": 2,
    }

    normalized = normalize_checkpoint_keys_for_model(model, state_dict)

    assert normalized == state_dict


def test_enforce_checkpoint_load_policy_raises_when_match_ratio_too_low():
    summary = {
        "checkpoint_path": "D:/tmp/best_model.pth",
        "model_key_count": 10,
        "checkpoint_key_count": 10,
        "matched_key_count": 1,
        "missing_key_count": 9,
        "unexpected_key_count": 9,
        "matched_key_ratio": 0.1,
        "missing_keys": ["a"],
        "unexpected_keys": ["b"],
    }

    try:
        enforce_checkpoint_load_policy(summary, init_label="warmup", min_match_ratio=0.5)
    except RuntimeError as exc:
        message = str(exc)
        assert "warmup" in message
        assert "0.1000" in message
        assert "D:/tmp/best_model.pth" in message
    else:
        raise AssertionError("Expected enforce_checkpoint_load_policy() to raise.")


def test_extract_resume_progress_reports_completed_and_next_epoch_from_checkpoint():
    checkpoint = {
        "epoch": 7,
        "global_step": 2800,
    }

    progress = extract_resume_progress(checkpoint)

    assert progress == {
        "lightning_completed_epochs": 7,
        "human_next_epoch": 8,
        "global_step": 2800,
    }


def test_build_resume_summary_message_includes_checkpoint_progress():
    checkpoint = {
        "epoch": 7,
        "global_step": 2800,
    }

    message = build_resume_summary_message(
        "D:/tmp/exp/last.ckpt", checkpoint, max_epochs=50
    )

    assert message == (
        "【断点接力】D:/tmp/exp/last.ckpt\n"
        "  检查点内 Lightning epoch=7（已完整结束 7 个训练 epoch）；"
        "续训进度条将从 Epoch 8/50 起接力至 50/50。"
        " global_step=2800"
    )


def test_build_resume_summary_message_returns_none_without_checkpoint_path():
    assert build_resume_summary_message(None, {"epoch": 7, "global_step": 2800}) is None


def test_resolve_trainer_runtime_config_uses_explicit_cpu_settings_when_cuda_unavailable():
    config = {"training": {"gpus": [0, 1]}}

    runtime = resolve_trainer_runtime_config(config, cuda_available=False)

    assert runtime == {
        "devices": 1,
        "accelerator": "cpu",
        "use_ddp": False,
        "sync_batchnorm": False,
        "precision": "32-true",
    }


def test_resolve_trainer_runtime_config_keeps_multi_gpu_ddp_when_cuda_available():
    config = {"training": {"gpus": [0, 1], "precision": "16-mixed"}}

    runtime = resolve_trainer_runtime_config(config, cuda_available=True)

    assert runtime == {
        "devices": [0, 1],
        "accelerator": "cuda",
        "use_ddp": True,
        "sync_batchnorm": True,
        "precision": "16-mixed",
    }


def test_resolve_datamodule_runtime_config_uses_cpu_safe_loader_settings_when_cuda_unavailable():
    config = {
        "datamodule": {
            "data_config": {
                "batch_size": 1,
                "num_workers": 4,
                "pin_memory": True,
                "persistent_workers": True,
                "segment": 4.0,
            }
        }
    }

    runtime = resolve_datamodule_runtime_config(config, cuda_available=False)

    assert runtime == {
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "segment": 4.0,
    }


def test_resolve_datamodule_runtime_config_preserves_loader_settings_when_cuda_available():
    config = {
        "datamodule": {
            "data_config": {
                "batch_size": 1,
                "num_workers": 4,
                "pin_memory": True,
                "persistent_workers": True,
            }
        }
    }

    runtime = resolve_datamodule_runtime_config(config, cuda_available=True)

    assert runtime == {
        "batch_size": 1,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
    }


def test_cleanup_training_runtime_releases_trainer_wandb_cuda_and_distributed_resources():
    trainer = types.SimpleNamespace(teardown=lambda: None)
    trainer.teardown = types.MethodType(lambda self: None, trainer)
    wandb_module = types.SimpleNamespace(finish=lambda: None)
    cuda_module = types.SimpleNamespace(empty_cache=lambda: None)
    distributed_module = types.SimpleNamespace(
        is_initialized=lambda: True,
        destroy_process_group=lambda: None,
    )

    calls = []
    trainer.teardown = lambda: calls.append("trainer")
    wandb_module.finish = lambda: calls.append("wandb")
    cuda_module.empty_cache = lambda: calls.append("cuda")
    distributed_module.destroy_process_group = lambda: calls.append("distributed")

    cleanup_training_runtime(
        trainer=trainer,
        wandb_module=wandb_module,
        cuda_module=cuda_module,
        distributed_module=distributed_module,
    )

    assert calls == ["trainer", "wandb", "cuda", "distributed"]


def test_cleanup_training_runtime_skips_distributed_teardown_when_process_group_is_not_initialized():
    distributed_module = types.SimpleNamespace(
        is_initialized=lambda: False,
        destroy_process_group=lambda: (_ for _ in ()).throw(
            AssertionError("destroy_process_group should not be called")
        ),
    )

    cleanup_training_runtime(
        trainer=None,
        wandb_module=types.SimpleNamespace(finish=lambda: None),
        cuda_module=types.SimpleNamespace(empty_cache=lambda: None),
        distributed_module=distributed_module,
    )
