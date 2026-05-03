"""BinaryDistillAudioLitModule 最小单测（蒸馏分支构建与损失组合）。"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
import torch.nn as nn

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

from look2hear.layers.kd_losses import Combined_KDLoss, SI_SNR_KDLoss, Subband_KDLoss
from look2hear.layers.binary_layers import BinaryConv1d, RPReLU, RSign
from look2hear.system.binary_distill_litmodule import BinaryDistillAudioLitModule


def _base_config(distillation: dict) -> dict:
    return {
        "datamodule": {"data_config": {"sample_rate": 16000}},
        "training": {
            "SpeedAug": False,
            "binary_stage_epochs": {"warmup": 1, "binary": 10},
        },
        "optimizer": {},
        "distillation": distillation,
    }


BAND_WIDTH_67 = [1] * 40 + [4] * 10 + [10] * 8 + [20] * 8 + [61]


class StudentWithBandWidth(torch.nn.Module):
    """模拟 BinaryTIGER：`audio_model.model.band_width`。"""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.band_width = BAND_WIDTH_67

    def forward(self, wav):
        b, t = wav.shape[0], wav.shape[-1]
        return torch.zeros(b, 2, t, device=wav.device, dtype=wav.dtype)


def test_binary_distill_disabled_skips_kd_modules():
    m = BinaryDistillAudioLitModule(
        audio_model=torch.nn.Identity(),
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-3}]),
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.0),
            "val": lambda est, tgt: torch.tensor(1.0),
        },
        config=_base_config({"enabled": False, "kd_type": "d3"}),
        teacher_model=None,
    )
    assert m.kd_loss_out is None
    assert m.kd_loss_band is None
    assert m.kd_loss_combined is None


def test_binary_distill_d1_builds_si_snr_kd():
    m = BinaryDistillAudioLitModule(
        audio_model=torch.nn.Identity(),
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-3}]),
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.0),
            "val": lambda est, tgt: torch.tensor(1.0),
        },
        config=_base_config({"enabled": True, "kd_type": "d1"}),
        teacher_model=None,
    )
    assert isinstance(m.kd_loss_out, SI_SNR_KDLoss)
    assert m.kd_loss_band is None
    assert m.kd_loss_combined is None


def test_binary_distill_d2_requires_band_width():
    with pytest.raises(ValueError, match="band_width"):
        BinaryDistillAudioLitModule(
            audio_model=torch.nn.Identity(),
            optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-3}]),
            loss_func={
                "train": lambda est, tgt: torch.tensor(1.0),
                "val": lambda est, tgt: torch.tensor(1.0),
            },
            config=_base_config({"enabled": True, "kd_type": "d2"}),
            teacher_model=None,
        )


def test_binary_distill_d2_d3_build_subband_losses():
    student = StudentWithBandWidth()
    m2 = BinaryDistillAudioLitModule(
        audio_model=student,
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-3}]),
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.0),
            "val": lambda est, tgt: torch.tensor(1.0),
        },
        config=_base_config({"enabled": True, "kd_type": "d2"}),
        teacher_model=None,
    )
    assert isinstance(m2.kd_loss_band, Subband_KDLoss)

    m3 = BinaryDistillAudioLitModule(
        audio_model=student,
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-3}]),
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.0),
            "val": lambda est, tgt: torch.tensor(1.0),
        },
        config=_base_config({"enabled": True, "kd_type": "d3"}),
        teacher_model=None,
    )
    assert isinstance(m3.kd_loss_combined, Combined_KDLoss)


def test_calibrate_loss_scales_clamped():
    m = BinaryDistillAudioLitModule(
        audio_model=torch.nn.Identity(),
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-3}]),
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.0),
            "val": lambda est, tgt: torch.tensor(1.0),
        },
        config=_base_config({"enabled": True, "kd_type": "d1"}),
        teacher_model=None,
    )
    m._calibrated = False
    m._calibrate_loss(torch.tensor(10.0), torch.tensor(0.0001), None)
    assert m._calib_scale_out == 100.0
    assert m._calibrated

    m2 = BinaryDistillAudioLitModule(
        audio_model=torch.nn.Identity(),
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-3}]),
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.0),
            "val": lambda est, tgt: torch.tensor(1.0),
        },
        config=_base_config({"enabled": True, "kd_type": "d1"}),
        teacher_model=None,
    )
    m2._calibrated = False
    m2._calibrate_loss(torch.tensor(0.01), torch.tensor(10.0), None)
    assert m2._calib_scale_out == 0.01


def test_cosine_decay_endpoints():
    start = BinaryDistillAudioLitModule._cosine_decay(0.0, 1.0, 0.1)
    end = BinaryDistillAudioLitModule._cosine_decay(1.0, 1.0, 0.1)
    assert abs(start - 1.0) < 1e-5
    assert abs(end - 0.1) < 1e-5


def test_distill_warmup_trains_only_rsign_and_rprelu():
    class DummyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.band_width = BAND_WIDTH_67
            self.binary = BinaryConv1d(1, 1, 1)
            self.protected = nn.Conv1d(1, 1, 1)
            self.rsign = RSign(1)
            self.rprelu = RPReLU(1)

        def forward(self, wav):
            b, t = wav.shape[0], wav.shape[-1]
            return torch.zeros(b, 2, t, device=wav.device, dtype=wav.dtype)

    student = DummyStudent()
    teacher = SmallSignalStudent()
    module = BinaryDistillAudioLitModule(
        audio_model=student,
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-3}]),
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.0),
            "val": lambda est, tgt: torch.tensor(1.0),
        },
        config=_base_config({"enabled": True, "kd_type": "d3", "distill_warmup_epochs": 5}),
        teacher_model=teacher,
    )

    with patch.object(type(module), "current_epoch", new_callable=PropertyMock) as current_epoch:
        current_epoch.return_value = 0
        module.on_train_epoch_start()

    params = dict(module.audio_model.named_parameters())
    assert params["binary.weight"].requires_grad is False
    assert params["protected.weight"].requires_grad is False
    assert params["rsign.alpha"].requires_grad is True
    assert params["rprelu.beta"].requires_grad is True


class SmallSignalStudent(torch.nn.Module):
    """避免全零波形导致 SI-SNR 数值不稳定。"""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.band_width = BAND_WIDTH_67

    def forward(self, wav):
        b, t = wav.shape[0], wav.shape[-1]
        return torch.ones(b, 2, t, device=wav.device, dtype=wav.dtype) * 1e-3


def test_distill_training_step_d1_returns_loss():
    student = SmallSignalStudent()
    teacher = SmallSignalStudent()

    m = BinaryDistillAudioLitModule(
        audio_model=student,
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 1e-3}]),
        loss_func={
            "train": lambda est, tgt: torch.tensor(2.0),
            "val": lambda est, tgt: torch.tensor(2.0),
        },
        config=_base_config(
            {
                "enabled": True,
                "kd_type": "d1",
                "loss_calibration": False,
            }
        ),
        teacher_model=teacher,
    )
    m.log = MagicMock()
    trainer = MagicMock()
    trainer.max_epochs = 10
    m.trainer = trainer

    b, t = 2, 4000
    mix = torch.randn(b, t)
    tgt = torch.randn(b, 2, t)
    batch = (mix, tgt, None)

    with patch.object(type(m), "current_epoch", new_callable=PropertyMock) as ce:
        ce.return_value = 0
        out = m.training_step(batch, 0)
    assert "loss" in out
    assert out["loss"].ndim == 0
    # task 2.0 + λ·L_kd；L_kd 为负 SI-SNR，和应为有限标量
    assert torch.isfinite(out["loss"]).item()
