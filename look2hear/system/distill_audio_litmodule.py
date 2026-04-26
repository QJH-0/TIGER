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
        # 兼容旧版：kd_lambda 作为混合系数 (1-kd)*task + kd*kd_loss
        self.kd_lambda = float(self.distill_config.get("kd_lambda", 0.0))
        # 新版：lambda_kd 直接加到 task_loss 上，并支持余弦衰减
        self.lambda_kd_schedule = str(self.distill_config.get("lambda_kd_schedule", "") or "")
        self.lambda_kd_range = self.distill_config.get("lambda_kd_range")
        self.dispatch_loss = DISPATCHLoss(
            n_fft=int(self.distill_config.get("n_fft", 640)),
            hop_length=int(self.distill_config.get("hop_length", 160)),
            patch_size=(
                int(self.distill_config["patch_size"])
                if "patch_size" in self.distill_config
                else None
            ),
            top_k_percent=float(self.distill_config.get("top_k_percent", 0.3)),
            low_freq_patch_size=(
                int(self.distill_config["low_freq_patch_size"])
                if "low_freq_patch_size" in self.distill_config
                else None
            ),
            high_freq_patch_size=(
                int(self.distill_config["high_freq_patch_size"])
                if "high_freq_patch_size" in self.distill_config
                else None
            ),
            split_freq_bin=(
                int(self.distill_config["split_freq_bin"])
                if "split_freq_bin" in self.distill_config
                else None
            ),
        )

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.teacher_model is not None:
            self._print_model_size_summary(
                model=self.teacher_model,
                label=f"Teacher:{type(self.teacher_model).__name__}",
            )

    @staticmethod
    def _cosine_decay(progress: float, start: float, end: float) -> float:
        # progress: 0..1
        p = float(min(max(progress, 0.0), 1.0))
        return float(end + 0.5 * (start - end) * (1.0 + torch.cos(torch.tensor(p * 3.141592653589793)).item()))

    def _resolve_lambda_kd(self) -> float | None:
        if not self.lambda_kd_schedule or not self.lambda_kd_range:
            return None
        if str(self.lambda_kd_schedule).lower() != "cosine":
            return None
        try:
            start, end = float(self.lambda_kd_range[0]), float(self.lambda_kd_range[1])
        except Exception:
            return None
        total = getattr(self.trainer, "max_epochs", None) or 0
        if total <= 1:
            progress = 1.0
        else:
            progress = float(self.current_epoch) / float(total - 1)
        return float(self._cosine_decay(progress, start, end))

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
            lambda_kd = self._resolve_lambda_kd()
            if lambda_kd is not None:
                loss = task_loss + float(lambda_kd) * kd_loss
                self.log("train/lambda_kd", float(lambda_kd), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, logger=True)
            else:
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
            self.log(
                "train/kgs_pos_ratio",
                stats.get("kgs_pos_ratio", 0.0),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                logger=True,
            )

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        return {"loss": loss}
