import sys
import types
from pathlib import Path
from unittest.mock import PropertyMock, patch

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
