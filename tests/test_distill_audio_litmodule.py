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


def test_on_fit_start_prints_student_and_teacher_model_stats():
    class DummyStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 3)

        def forward(self, mixtures):
            return mixtures

    class DummyTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 2, bias=False)

        def forward(self, mixtures):
            return mixtures

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
            "distillation": {"enabled": True},
        },
        teacher_model=DummyTeacher(),
    )
    messages = []
    module.print = lambda message: messages.append(message)

    module.on_fit_start()

    assert any("[DummyStudent] params total=15 trainable=15 frozen=0" in message for message in messages)
    assert any("[Teacher:DummyTeacher] params total=8 trainable=8 frozen=0" in message for message in messages)
