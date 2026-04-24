import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import MutableMapping
from speechbrain.processing.speech_augmentation import SpeedPerturb

def flatten_dict(d, parent_key="", sep="_"):
    """Flattens a dictionary into a single-level dictionary while preserving
    parent keys. Taken from
    `SO <https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys>`_

    Args:
        d (MutableMapping): Dictionary to be flattened.
        parent_key (str): String to use as a prefix to all subsequent keys.
        sep (str): String to use as a separator between two key levels.

    Returns:
        dict: Single-level dictionary, flattened.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class AudioLightningModule(pl.LightningModule):
    def __init__(
        self,
        audio_model=None,
        video_model=None,
        optimizer=None,
        loss_func=None,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config
        # Speed Aug
        self.speedperturb = SpeedPerturb(
            self.config["datamodule"]["data_config"]["sample_rate"],
            speeds=[95, 100, 105],
            perturb_prob=1.0
        )
        # Save lightning"s AttributeDict under self.hparams
        self.default_monitor = "val/loss"
        self.save_hyperparameters(self.config_to_hparams(self.config))
        # self.print(self.audio_model)
        self.validation_step_outputs = []
        self._printed_model_size_summary = False

    def _normalize_scheduler_monitor(self, monitor):
        # 兼容旧配置中遗留的 monitor 命名。
        if monitor in {"val/loss", "val/loss_epoch", "val/loss/dataloader_idx_0"}:
            return self.default_monitor
        return monitor

    def _epoch_display_1based(self) -> float:
        """与自定义 Rich 进度条一致：第 1..max_epochs 轮（Lightning 内部仍为接力计数）。"""
        tr = self._trainer
        ce = self.current_epoch
        if tr is None or tr.max_epochs is None:
            return float(ce + 1)
        return float(min(ce + 1, tr.max_epochs))

    def forward(self, wav, mouth=None):
        """Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        return self.audio_model(wav)

    def training_step(self, batch, batch_nb):
        mixtures, targets, _ = batch
        
        new_targets = []
        min_len = -1
        if self.config["training"]["SpeedAug"] == True:
            with torch.no_grad():
                for i in range(targets.shape[1]):
                    new_target = self.speedperturb(targets[:, i, :])
                    new_targets.append(new_target)
                    if i == 0:
                        min_len = new_target.shape[-1]
                    else:
                        if new_target.shape[-1] < min_len:
                            min_len = new_target.shape[-1]

                targets = torch.zeros(
                            targets.shape[0],
                            targets.shape[1],
                            min_len,
                            device=targets.device,
                            dtype=torch.float,
                        )
                for i, new_target in enumerate(new_targets):
                    targets[:, i, :] = new_targets[i][:, 0:min_len]
                    
                mixtures = targets.sum(1)
        # print(mixtures.shape)
        est_sources = self(mixtures)
        loss = self.loss_func["train"](est_sources, targets)

        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        mixtures, targets, _ = batch
        est_sources = self(mixtures)
        loss = self.loss_func["val"](est_sources, targets)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

        self.validation_step_outputs.append(loss)

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # val
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        val_loss = torch.mean(self.all_gather(avg_loss))
        self.log(
            "epoch",
            self._epoch_display_1based(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "train/learning_rate",
            self.optimizer.param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "val/si_snr",
            -val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                # 兼容旧配置里未带 dataloader 后缀的 monitor，避免 ReduceLROnPlateau 在 epoch 结束时报错。
                sched["monitor"] = self._normalize_scheduler_monitor(
                    sched.get("monitor", self.default_monitor)
                )
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     if metric is None:
    #         scheduler.step()
    #     else:
    #         scheduler.step(metric)
    
    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                # 仅把纯数值序列转成 Tensor；字符串列表等配置保持原始形态。
                if all(isinstance(item, (bool, int, float)) for item in v):
                    dic[k] = torch.tensor(v)
        return dic

    @staticmethod
    def _model_size_summary(model):
        total_params = sum(parameter.numel() for parameter in model.parameters())
        trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        frozen_params = total_params - trainable_params
        actual_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        fp32_estimated_bytes = total_params * 4
        return {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "frozen_params": int(frozen_params),
            "actual_param_bytes": int(actual_bytes),
            "actual_param_size_mb": round(actual_bytes / (1024 ** 2), 4),
            "fp32_estimated_bytes": int(fp32_estimated_bytes),
            "fp32_estimated_size_mb": round(fp32_estimated_bytes / (1024 ** 2), 4),
        }

    def _print_model_size_summary(self, model=None, label=None):
        model = self.audio_model if model is None else model
        if model is None:
            return

        summary = self._model_size_summary(model)
        label = label or type(model).__name__
        self.print(
            f"[{label}] params total={summary['total_params']} "
            f"trainable={summary['trainable_params']} frozen={summary['frozen_params']}"
        )
        self.print(
            f"[{label}] size_mb fp32_estimated={summary['fp32_estimated_size_mb']} "
            f"actual={summary['actual_param_size_mb']}"
        )

    def on_fit_start(self) -> None:
        if self._printed_model_size_summary:
            return
        self._print_model_size_summary()
        self._printed_model_size_summary = True
