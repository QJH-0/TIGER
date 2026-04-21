import sys
import types
from pathlib import Path

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

from look2hear.system.distill_audio_litmodule import DistillAudioLightningModule


def test_distill_module_accepts_teacher():
    module = DistillAudioLightningModule(
        audio_model=torch.nn.Identity(),
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-3}]),
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.0),
            "val": lambda est, tgt: torch.tensor(1.0),
        },
        config={
            "datamodule": {"data_config": {"sample_rate": 16000}},
            "training": {"SpeedAug": False},
            "distillation": {
                "enabled": True,
                "kd_lambda": 0.3,
                "n_fft": 320,
                "hop_length": 160,
                "patch_size": 4,
                "top_k_percent": 0.25,
            },
        },
    )
    module.teacher_model = torch.nn.Identity()
    assert module.teacher_model is not None


def test_distill_module_logs_task_and_kd_loss_separately():
    class DummyStudent(torch.nn.Module):
        def forward(self, mixtures):
            return mixtures.unsqueeze(1).repeat(1, 2, 1)

    class DummyTeacher(torch.nn.Module):
        def forward(self, mixtures):
            return mixtures.unsqueeze(1).repeat(1, 2, 1) * 0.5

    module = DistillAudioLightningModule(
        audio_model=DummyStudent(),
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-3}]),
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.0),
            "val": lambda est, tgt: torch.tensor(1.0),
        },
        config={
            "datamodule": {"data_config": {"sample_rate": 16000}},
            "training": {"SpeedAug": False},
            "distillation": {
                "enabled": True,
                "kd_lambda": 0.3,
                "n_fft": 320,
                "hop_length": 160,
                "patch_size": 4,
                "top_k_percent": 0.25,
            },
        },
        teacher_model=DummyTeacher(),
    )
    logged = []
    module.log = lambda name, value, **kwargs: logged.append(name)
    batch = (
        torch.randn(2, 3200),
        torch.randn(2, 2, 3200),
        None,
    )

    result = module.training_step(batch, 0)

    assert "train/task_loss" in logged
    assert "train/kd_loss" in logged
    assert "train/loss" in logged
    assert "loss" in result
