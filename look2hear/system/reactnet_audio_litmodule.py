from __future__ import annotations

import torch

from ..layers.distillation_losses import DistributionLoss
from .binary_audio_litmodule import BinaryAudioLightningModule


class ReactNetAudioLightningModule(BinaryAudioLightningModule):
    """ReActNet 两阶段训练系统（激活二值化 warmup -> 全二值化 + 分布对齐衰减）。"""

    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

        reactnet_cfg = self.config.get("reactnet", {}) or {}
        self.step1_epochs = int(reactnet_cfg.get("step1_epochs", 0))
        self.step2_epochs = int(reactnet_cfg.get("step2_epochs", 0))
        self.step1_dist_weight = float(reactnet_cfg.get("step1_dist_weight", 0.1))
        self.dist_temp = float(reactnet_cfg.get("dist_temp", 1.0))

        self.dist_loss = DistributionLoss(temp=self.dist_temp)

    def _resolve_stage(self) -> str:
        # 按方案定义阶段：
        # - Step 1：仅激活二值化（权重全精度），禁用 RPReLU
        # - Step 2：权重 + 激活全二值化，启用 RPReLU
        # - Step 2 结束后若继续训练，进入 finetune（等价于保持全二值化）
        if self.step1_epochs <= 0 and self.step2_epochs <= 0:
            return super()._resolve_stage()

        if self.current_epoch < self.step1_epochs:
            return "activation_warmup"

        if self.current_epoch < self.step1_epochs + max(self.step2_epochs, 0):
            return "weight_binarize"

        return "finetune"

    def _resolve_dist_weight(self) -> float:
        if self.step1_epochs <= 0 and self.step2_epochs <= 0:
            return 0.0

        if self.current_epoch < self.step1_epochs:
            return float(self.step1_dist_weight)

        if self.step2_epochs <= 0:
            return 0.0

        progress = float(self.current_epoch - self.step1_epochs) / float(self.step2_epochs)
        return float(max(0.0, 1.0 - progress))

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.teacher_model is not None:
            self._print_model_size_summary(
                model=self.teacher_model,
                label=f"Teacher:{type(self.teacher_model).__name__}",
            )

    def training_step(self, batch, batch_nb):
        # 仍复用 BinaryAudioLightningModule 的阶段切换（binary on/off + RPReLU active）。
        stage = self._apply_stage()

        mixtures, targets, _ = batch
        est_sources = self(mixtures)

        task_loss = self.loss_func["train"](est_sources, targets)
        loss = task_loss

        self.log(
            "train/task_loss",
            task_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            logger=True,
        )
        self.log(
            "train/stage",
            float({"activation_warmup": 1.0, "weight_binarize": 2.0, "finetune": 3.0}.get(stage, 0.0)),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            logger=True,
        )

        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_sources = self.teacher_model(mixtures)
            dist_w = self._resolve_dist_weight()
            dist_loss = self.dist_loss(est_sources, teacher_sources)
            loss = task_loss + float(dist_w) * dist_loss

            self.log(
                "train/dist_loss",
                dist_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                logger=True,
            )
            self.log(
                "train/dist_weight",
                float(dist_w),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                logger=True,
            )

        # 二值阶段维持 clamp（与 BinaryAudioLightningModule 保持一致）
        if stage in {"binary", "weight_binarize", "finetune"} and hasattr(
            self.audio_model, "clamp_all_binary_weights"
        ):
            self.audio_model.clamp_all_binary_weights()

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

