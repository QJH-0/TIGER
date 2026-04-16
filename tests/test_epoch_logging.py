import sys
import types
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

from look2hear.system.audio_litmodule import AudioLightningModule
from audio_train import configure_wandb_epoch_metrics


def test_training_step_logs_epoch_only_train_metric_name():
    module = AudioLightningModule(
        audio_model=lambda wav: wav.unsqueeze(1).repeat(1, 2, 1),
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 0.001}]),
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.5),
            "val": lambda est, tgt: torch.tensor(2.5),
        },
        config={
            "datamodule": {"data_config": {"sample_rate": 16000}},
            "training": {"SpeedAug": False},
        },
    )
    logged = []
    module.log = lambda name, value, **kwargs: logged.append((name, kwargs))

    batch = (
        torch.randn(2, 8),
        torch.randn(2, 2, 8),
        None,
    )
    module.training_step(batch, 0)

    assert ("train/loss_epoch", {"on_step": False, "on_epoch": True, "prog_bar": True, "sync_dist": True, "logger": True}) in logged


def test_validation_epoch_end_avoids_manual_wandb_logging():
    module = AudioLightningModule(
        audio_model=lambda wav: wav.unsqueeze(1).repeat(1, 2, 1),
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 0.001}]),
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.5),
            "val": lambda est, tgt: torch.tensor(2.5),
        },
        config={
            "datamodule": {"data_config": {"sample_rate": 16000}},
            "training": {"SpeedAug": False},
        },
    )
    logged = []
    module.log = lambda name, value, **kwargs: logged.append((name, kwargs))
    experiment = Mock()
    module.validation_step_outputs = [torch.tensor(2.0), torch.tensor(4.0)]
    module.test_step_outputs = [torch.tensor(3.0)]
    module.all_gather = lambda value: value

    with patch.object(type(module), "logger", new_callable=PropertyMock) as logger_prop:
        logger_prop.return_value = types.SimpleNamespace(experiment=experiment)
        module.on_validation_epoch_end()

    names = [name for name, _ in logged]
    assert "val/pit_sisnr_epoch" in names
    assert "test/pit_sisnr_epoch" in names
    assert "train/lr_epoch" in names
    experiment.log.assert_not_called()


def test_configure_wandb_epoch_metrics_uses_epoch_as_step_metric():
    experiment = Mock()
    logger = types.SimpleNamespace(experiment=experiment)

    configure_wandb_epoch_metrics(logger)

    experiment.define_metric.assert_any_call("epoch")
    experiment.define_metric.assert_any_call("train/*", step_metric="epoch")
    experiment.define_metric.assert_any_call("val/*", step_metric="epoch")
    experiment.define_metric.assert_any_call("test/*", step_metric="epoch")


def test_reduce_on_plateau_uses_validation_metric_with_dataloader_suffix():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = ReduceLROnPlateau(optimizer, patience=1)
    module = AudioLightningModule(
        audio_model=lambda wav: wav.unsqueeze(1).repeat(1, 2, 1),
        optimizer=optimizer,
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.5),
            "val": lambda est, tgt: torch.tensor(2.5),
        },
        scheduler=scheduler,
        config={
            "datamodule": {"data_config": {"sample_rate": 16000}},
            "training": {"SpeedAug": False},
        },
    )

    optimizers, schedulers = module.configure_optimizers()

    assert optimizers == [optimizer]
    assert schedulers[0]["monitor"] == "val/loss_epoch/dataloader_idx_0"
