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
from audio_train import (
    build_wandb_project_name,
    build_wandb_run_name,
    configure_wandb_epoch_metrics,
    sanitize_wandb_config,
)


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

    assert ("train/loss", {"on_step": False, "on_epoch": True, "prog_bar": True, "sync_dist": True, "logger": True}) in logged


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
    module.all_gather = lambda value: value

    with patch.object(type(module), "logger", new_callable=PropertyMock) as logger_prop:
        logger_prop.return_value = types.SimpleNamespace(experiment=experiment)
        module.on_validation_epoch_end()

    names = [name for name, _ in logged]
    assert "val/si_snr" in names
    assert "train/learning_rate" in names
    assert all(not name.startswith("test/") for name in names)
    experiment.log.assert_not_called()


def test_configure_wandb_epoch_metrics_uses_epoch_as_step_metric():
    experiment = Mock()
    logger = types.SimpleNamespace(experiment=experiment)

    configure_wandb_epoch_metrics(logger)

    experiment.define_metric.assert_any_call("epoch", hidden=True)
    experiment.define_metric.assert_any_call("trainer/global_step", hidden=True)
    experiment.define_metric.assert_any_call("train/loss", step_metric="epoch")
    experiment.define_metric.assert_any_call("train/learning_rate", step_metric="epoch")
    experiment.define_metric.assert_any_call("val/loss", step_metric="epoch")
    experiment.define_metric.assert_any_call("val/si_snr", step_metric="epoch")
    defined_metrics = [call.args[0] for call in experiment.define_metric.call_args_list]
    assert "train/*" not in defined_metrics
    assert "val/*" not in defined_metrics
    assert "test/*" not in defined_metrics


def test_sanitize_wandb_config_flattens_nested_keys_with_underscores():
    config = {
        "training": {"epochs": 5},
        "datamodule": {"data_config": {"batch_size": 4, "segment": 3.0}},
        "scheduler": {"sche_name": None},
    }

    flat_config = sanitize_wandb_config(config)

    assert flat_config["training_epochs"] == 5
    assert flat_config["datamodule_data_config_batch_size"] == 4
    assert flat_config["datamodule_data_config_segment"] == 3.0
    assert flat_config["scheduler_sche_name"] == "None"
    assert all("." not in key for key in flat_config)


def test_build_wandb_project_name_is_stable():
    config = {"exp": {"exp_name": "TIGER-MiniLibriMix"}}

    assert build_wandb_project_name(config) == "tiger-speech-separation"


def test_build_wandb_run_name_uses_model_dataset_and_core_hparams():
    config = {
        "audionet": {"audionet_name": "TIGER"},
        "datamodule": {
            "data_name": "Libri2MixModuleRemix",
            "data_config": {"batch_size": 4, "segment": 3.0},
        },
        "exp": {"exp_name": "Kaggle-T4x2"},
    }

    run_name = build_wandb_run_name(config)

    assert run_name == "tiger-libri2mixmoduleremix-bs4-seg3.0-kaggle-t4x2"


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
    assert schedulers[0]["monitor"] == "val/loss"


def test_scheduler_dict_rewrites_legacy_validation_monitor_name():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = {
        "scheduler": ReduceLROnPlateau(optimizer, patience=1),
        "monitor": "val/loss",
        "interval": "epoch",
    }
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
    assert schedulers[0]["monitor"] == "val/loss"


def test_scheduler_dict_rewrites_dataloader_suffix_monitor_to_plain_validation_metric():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = {
        "scheduler": ReduceLROnPlateau(optimizer, patience=1),
        "monitor": "val/loss/dataloader_idx_0",
        "interval": "epoch",
    }
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
    assert schedulers[0]["monitor"] == "val/loss"


def test_validation_dataloader_only_returns_validation_loader():
    val_loader = object()
    test_loader = object()
    module = AudioLightningModule(
        audio_model=lambda wav: wav.unsqueeze(1).repeat(1, 2, 1),
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 0.001}]),
        loss_func={
            "train": lambda est, tgt: torch.tensor(1.5),
            "val": lambda est, tgt: torch.tensor(2.5),
        },
        val_loader=val_loader,
        test_loader=test_loader,
        config={
            "datamodule": {"data_config": {"sample_rate": 16000}},
            "training": {"SpeedAug": False},
        },
    )

    assert module.val_dataloader() is val_loader
