import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

speechbrain = types.ModuleType("speechbrain")
processing = types.ModuleType("speechbrain.processing")
speech_augmentation = types.ModuleType("speechbrain.processing.speech_augmentation")


class DummySpeedPerturb:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, tensor):
        return tensor


speech_augmentation.SpeedPerturb = DummySpeedPerturb
processing.speech_augmentation = speech_augmentation
speechbrain.processing = processing

sys.modules.setdefault("speechbrain", speechbrain)
sys.modules.setdefault("speechbrain.processing", processing)
sys.modules.setdefault("speechbrain.processing.speech_augmentation", speech_augmentation)

from look2hear.system.binary_audio_litmodule import BinaryAudioLightningModule


def test_resolve_binary_stage():
    module = BinaryAudioLightningModule(
        audio_model=torch.nn.Identity(),
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-3}]),
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.0),
            "val": lambda est, tgt: torch.tensor(1.0),
        },
        config={
            "datamodule": {"data_config": {"sample_rate": 16000}},
            "training": {"SpeedAug": False, "binary_stage_epochs": {"warmup": 2, "binary": 3}},
        },
    )
    with patch.object(type(module), "current_epoch", new_callable=PropertyMock) as current_epoch:
        current_epoch.return_value = 0
        assert module._resolve_stage() == "warmup"


def test_binary_module_supports_three_stage_schedule():
    module = BinaryAudioLightningModule(
        audio_model=torch.nn.Identity(),
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-3}]),
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.0),
            "val": lambda est, tgt: torch.tensor(1.0),
        },
        config={
            "datamodule": {"data_config": {"sample_rate": 16000}},
            "training": {
                "SpeedAug": False,
                "binary_stage_epochs": {"activation_warmup": 1, "weight_binarize": 3},
            },
        },
    )
    with patch.object(type(module), "current_epoch", new_callable=PropertyMock) as current_epoch:
        current_epoch.return_value = 0
        assert module._resolve_stage() == "activation_warmup"
        current_epoch.return_value = 1
        assert module._resolve_stage() == "weight_binarize"
        current_epoch.return_value = 3
        assert module._resolve_stage() == "finetune"


def test_on_fit_start_prints_binarization_summary():
    class DummyBinaryModel(torch.nn.Module):
        def get_binarization_summary(self):
            return {
                "binary_params": 20,
                "total_params": 100,
                "binary_ratio": 0.2,
                "binary_module_names": ["model.layer1.conv", "model.layer2.conv"],
                "protected_module_names": ["model.mask.0.1"],
            }

        def set_binary_training(self, enabled: bool):
            self.enabled = enabled

        def clamp_all_binary_weights(self):
            self.clamped = True

    module = BinaryAudioLightningModule(
        audio_model=DummyBinaryModel(),
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-3}]),
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.0),
            "val": lambda est, tgt: torch.tensor(1.0),
        },
        config={
            "datamodule": {"data_config": {"sample_rate": 16000}},
            "training": {"SpeedAug": False, "binary_stage_epochs": {"warmup": 2, "binary": 3}},
        },
    )
    messages = []
    module.print = lambda message: messages.append(message)

    module.on_fit_start()

    assert any("binary_params=20/100" in message for message in messages)
    assert any("model.layer1.conv" in message for message in messages)


def test_maybe_update_binary_ema_scales_respects_training_flag():
    """EMA 仅在 training=True 且模型实现 update_all_ema_scales 时调用。"""

    class DummyEma(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def update_all_ema_scales(self):
            self.calls += 1

    dummy = DummyEma()
    module = BinaryAudioLightningModule(
        audio_model=dummy,
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-3}]),
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.0),
            "val": lambda est, tgt: torch.tensor(1.0),
        },
        config={
            "datamodule": {"data_config": {"sample_rate": 16000}},
            "training": {"SpeedAug": False, "binary_stage_epochs": {"warmup": 0, "binary": 1}},
        },
    )
    module.training = False
    module._maybe_update_binary_ema_scales()
    assert dummy.calls == 0
    module.training = True
    module._maybe_update_binary_ema_scales()
    assert dummy.calls == 1


def test_on_fit_end_writes_final_metrics_json(tmp_path):
    """训练结束在实验目录写入 final_metrics.json。"""
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir(parents=True, exist_ok=True)
    module = BinaryAudioLightningModule(
        audio_model=torch.nn.Linear(2, 2),
        optimizer=torch.optim.SGD(torch.nn.Linear(2, 2).parameters(), lr=0.1),
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.0),
            "val": lambda est, tgt: torch.tensor(1.0),
        },
        config={
            "datamodule": {"data_config": {"sample_rate": 16000}},
            "training": {
                "SpeedAug": False,
                "binary_stage_epochs": {"warmup": 0, "binary": 1},
                "freeze_scope": ["bn"],
                "write_final_metrics": True,
            },
            "exp": {"exp_name": "unit-test-exp"},
            "main_args": {"exp_dir": str(exp_dir)},
        },
    )
    trainer = MagicMock()
    trainer.max_epochs = 3
    trainer.callback_metrics = {"val/loss": torch.tensor(0.25), "train/loss": torch.tensor(0.5)}
    trainer.callbacks = []
    module.trainer = trainer
    module.print = lambda *_args, **_kwargs: None
    with patch.object(type(module), "current_epoch", new_callable=PropertyMock) as ce:
        ce.return_value = 2
        module.on_fit_end()
    out = exp_dir / "final_metrics.json"
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["config"]["exp_name"] == "unit-test-exp"
    assert data["config"]["freeze_scope"] == ["bn"]
    assert data["metrics"]["val_loss"] == 0.25
    assert data["config"]["total_epochs_completed"] == 3


def test_amp_compatible_prelu_handles_half_input_on_cuda():
    if not torch.cuda.is_available():
        return

    module = BinaryAudioLightningModule(
        audio_model=torch.nn.Identity(),
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-3}]),
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.0),
            "val": lambda est, tgt: torch.tensor(1.0),
        },
        config={
            "datamodule": {"data_config": {"sample_rate": 16000}},
            "training": {
                "SpeedAug": False,
                "binary_stage_epochs": {"warmup": 0, "binary": 1},
                "ablation": {"use_original_prelu": True},
            },
        },
    )
    module.audio_model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
    setattr(module.audio_model, "0", torch.nn.Identity())
    setattr(module.audio_model, "1", torch.nn.Identity())

    from look2hear.layers.binary_layers import RPReLU

    module.audio_model = torch.nn.Sequential(RPReLU(1)).cuda()
    module._apply_ablation()
    x = torch.randn(2, 1, 8, device="cuda", dtype=torch.float16)
    y = module.audio_model(x)
    assert y.dtype == torch.float16
