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

from look2hear.system.reactnet_audio_litmodule import ReactNetAudioLightningModule


def test_reactnet_module_supports_teacher_and_logs_distill_terms():
    class DummyStudent(torch.nn.Module):
        def forward(self, mixtures):
            return mixtures.unsqueeze(1).repeat(1, 2, 1)

    class DummyTeacher(torch.nn.Module):
        def forward(self, mixtures):
            return mixtures.unsqueeze(1).repeat(1, 2, 1) * 0.5

    module = ReactNetAudioLightningModule(
        audio_model=DummyStudent(),
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-3}]),
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.0),
            "val": lambda est, tgt: torch.tensor(1.0),
        },
        config={
            "datamodule": {"data_config": {"sample_rate": 16000}},
            "training": {"SpeedAug": False, "binary_stage_epochs": {}},
            "reactnet": {"step1_epochs": 2, "step2_epochs": 3, "step1_dist_weight": 0.1},
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
    assert "train/dist_loss" in logged
    assert "train/dist_weight" in logged
    assert "train/loss" in logged
    assert "loss" in result

