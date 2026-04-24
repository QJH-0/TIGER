from __future__ import annotations

import torch

from ..layers.dispatch_layers import DISPATCHLoss
from .audio_litmodule import AudioLightningModule


class DistillAudioLightningModule(AudioLightningModule):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distill_config = self.config.get("distillation", {})
        self.distillation_enabled = bool(self.distill_config.get("enabled", False))
        self.kd_lambda = float(self.distill_config.get("kd_lambda", 0.0))
        self.dispatch_loss = DISPATCHLoss(
            n_fft=int(self.distill_config.get("n_fft", 640)),
            hop_length=int(self.distill_config.get("hop_length", 160)),
            patch_size=int(self.distill_config.get("patch_size", 16)),
            top_k_percent=float(self.distill_config.get("top_k_percent", 0.3)),
        )

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.teacher_model is not None:
            self._print_model_size_summary(
                model=self.teacher_model,
                label=f"Teacher:{type(self.teacher_model).__name__}",
            )

    def training_step(self, batch, batch_nb):
        mixtures, targets, _ = batch
        est_sources = self(mixtures)
        task_loss = self.loss_func["train"](est_sources, targets)
        loss = task_loss
        # 独立记录任务损失，便于和蒸馏损失拆开观察。
        self.log("train/task_loss", task_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, logger=True)

        if self.distillation_enabled and self.teacher_model is not None:
            with torch.no_grad():
                teacher_sources = self.teacher_model(mixtures)
            kd_loss, stats = self.dispatch_loss(est_sources, teacher_sources, targets)
            loss = (1.0 - self.kd_lambda) * task_loss + self.kd_lambda * kd_loss
            self.log("train/kd_loss", kd_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, logger=True)
            self.log(
                "train/selected_patches_ratio",
                stats["selected_patches_ratio"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                logger=True,
            )

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        return {"loss": loss}
