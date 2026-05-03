"""统一的二值化 + 蒸馏训练系统。

通过配置灵活选择：
- 纯二值化训练（无蒸馏）
- D1: SI-SNR 输出蒸馏
- D2: 子带选择性蒸馏
- D3: 输出 + 子带联合蒸馏

支持：
- 参数分组（原始权重 lr / 二值化新增参数 lr）
- 蒸馏预热（冻结 BinaryConv 权重，仅训练 RSign/RPReLU）
- 损失校准（首 batch 自动 scale 蒸馏损失量级）
- 余弦 lambda 调度
- EMA Scale 更新（继承 ``BinaryAudioLightningModule.optimizer_step``，在优化步之后执行）
"""

from __future__ import annotations

from typing import Optional

import torch

from .binary_audio_litmodule import BinaryAudioLightningModule
from ..layers.binary_layers import BinaryConv1d, BinaryConv2d, BinaryLinear, RSign, RPReLU
from ..layers.kd_losses import SI_SNR_KDLoss, Subband_KDLoss, Combined_KDLoss


class BinaryDistillAudioLitModule(BinaryAudioLightningModule):
    """统一的二值化 + 蒸馏训练系统。

    继承 BinaryAudioLightningModule 的阶段管理（warmup -> binary -> finetune），
    在此基础上集成知识蒸馏（D1/D2/D3）。
    """

    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distill_config = self.config.get("distillation", {})
        self.distillation_enabled = bool(self.distill_config.get("enabled", False))

        # 蒸馏类型：d1 / d2 / d3
        self.kd_type = str(self.distill_config.get("kd_type", "d3")).lower()

        # lambda 调度
        self.lambda_out_range = self.distill_config.get("lambda_out_range", [1.0, 0.1])
        self.lambda_band_range = self.distill_config.get("lambda_band_range", [1.0, 0.1])

        # 蒸馏预热：冻结 BinaryConv 的 epoch 数
        self.distill_warmup_epochs = int(self.distill_config.get("distill_warmup_epochs", 5))

        # 损失校准
        self.loss_calibration_enabled = bool(self.distill_config.get("loss_calibration", True))
        self._calibrated = False
        self._calib_scale_out = 1.0
        self._calib_scale_band = 1.0

        # 初始化蒸馏损失
        self._build_kd_losses()

        # 参数分组配置
        self.param_group_config = self.config.get("optimizer", {}).get("param_groups", {})

    def _build_kd_losses(self) -> None:
        """根据配置构建蒸馏损失模块。"""
        if not self.distillation_enabled:
            self.kd_loss_out = None
            self.kd_loss_band = None
            self.kd_loss_combined = None
            return

        # 获取 TIGER 的 band_width（从 audio_model 中读取）
        tiger_model = self.audio_model
        if hasattr(tiger_model, "model"):
            tiger_model = tiger_model.model
        band_width = getattr(tiger_model, "band_width", None)

        n_fft = int(self.distill_config.get("n_fft", 640))
        hop_length = int(self.distill_config.get("hop_length", 160))
        weight_kwargs = {
            "low_freq_weight": float(self.distill_config.get("low_freq_weight", 2.0)),
            "mid_low_weight": float(self.distill_config.get("mid_low_weight", 1.5)),
            "mid_weight": float(self.distill_config.get("mid_weight", 1.0)),
            "mid_high_weight": float(self.distill_config.get("mid_high_weight", 0.8)),
            "high_freq_weight": float(self.distill_config.get("high_freq_weight", 0.5)),
        }

        if self.kd_type == "d1":
            self.kd_loss_out = SI_SNR_KDLoss()
            self.kd_loss_band = None
            self.kd_loss_combined = None
        elif self.kd_type == "d2":
            self.kd_loss_out = None
            if band_width is None:
                raise ValueError("D2 蒸馏需要 TIGER 的 band_width 属性")
            self.kd_loss_band = Subband_KDLoss(
                band_width=band_width,
                n_fft=n_fft,
                hop_length=hop_length,
                **weight_kwargs,
            )
            self.kd_loss_combined = None
        elif self.kd_type == "d3":
            # D3 使用 Combined_KDLoss 共享 PIT 排列
            self.kd_loss_out = None
            self.kd_loss_band = None
            if band_width is None:
                raise ValueError("D3 蒸馏需要 TIGER 的 band_width 属性")
            self.kd_loss_combined = Combined_KDLoss(
                band_width=band_width,
                n_fft=n_fft,
                hop_length=hop_length,
                **weight_kwargs,
            )
        else:
            raise ValueError(f"未知的蒸馏类型: {self.kd_type}，支持 d1/d2/d3")

    # ------------------------------------------------------------------
    #  参数分组
    # ------------------------------------------------------------------

    def _build_param_groups(self) -> list[dict]:
        """构建参数分组：原始权重用小 lr，二值化新增参数用大 lr。"""
        if not self.param_group_config:
            return None  # 不分组，使用默认

        fp32_lr = float(self.param_group_config.get("fp32_lr", 5e-5))
        binary_lr = float(self.param_group_config.get("binary_lr", 1e-3))

        fp32_params = []
        binary_params = []

        for name, param in self.audio_model.named_parameters():
            if not param.requires_grad:
                continue
            # 判断是否属于二值化新增参数
            is_binary_param = False
            # 检查是否在 BinaryConv1d/BinaryLinear 的新增参数中
            for module_name, module in self.audio_model.named_modules():
                if name.startswith(module_name + "."):
                    if isinstance(module, (BinaryConv1d, BinaryConv2d, BinaryLinear)):
                        # BinaryConv1d/BinaryLinear 的 weight/bias 是原始参数（需微调）
                        # 没有额外新增参数（scale 是 buffer，不在此列）
                        is_binary_param = False
                    elif isinstance(module, (RSign, RPReLU)):
                        is_binary_param = True
            if is_binary_param:
                binary_params.append(param)
            else:
                fp32_params.append(param)

        return [
            {"params": fp32_params, "lr": fp32_lr, "name": "fp32_weights"},
            {"params": binary_params, "lr": binary_lr, "name": "binary_new_params"},
        ]

    def configure_optimizers(self):
        """覆盖优化器配置，支持参数分组。

        当使用参数分组时，需要同时重建 scheduler，因为原 scheduler 绑定的是旧 optimizer。
        """
        param_groups = self._build_param_groups()
        if param_groups is not None:
            # 用参数分组替换原始 optimizer
            optim_config = self.config.get("optimizer", {})
            optim_name = str(optim_config.get("optim_name", "adamw")).lower()
            weight_decay = float(optim_config.get("weight_decay", 1e-4))

            from .optimizers import make_optimizer
            self.optimizer = make_optimizer(
                param_groups,
                optim_name=optim_name,
                weight_decay=weight_decay,
            )
            # 用新 optimizer 重建 scheduler，避免旧 scheduler 绑定旧 optimizer
            sche_config = self.config.get("scheduler", {})
            sche_name = sche_config.get("sche_name")
            if sche_name:
                import torch.optim.lr_scheduler as lr_schedulers
                sche_cls = getattr(lr_schedulers, sche_name, None)
                if sche_cls is not None:
                    self.scheduler = sche_cls(
                        self.optimizer, **sche_config.get("sche_config", {})
                    )
        return super().configure_optimizers()

    # ------------------------------------------------------------------
    #  蒸馏预热：冻结 BinaryConv
    # ------------------------------------------------------------------

    def _apply_distill_warmup(self) -> None:
        """Apply the paper-aligned distillation warmup."""
        is_warming_up = self.current_epoch < self.distill_warmup_epochs
        if not is_warming_up:
            return

        for _, param in self.audio_model.named_parameters():
            param.requires_grad = False

        for module in self.audio_model.modules():
            if isinstance(module, (RSign, RPReLU)):
                for param in module.parameters():
                    param.requires_grad = True

    # ------------------------------------------------------------------
    #  lambda 调度
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_decay(progress: float, start: float, end: float) -> float:
        p = float(min(max(progress, 0.0), 1.0))
        return float(end + 0.5 * (start - end) * (1.0 + torch.cos(torch.tensor(p * 3.141592653589793)).item()))

    def _resolve_lambda(self, range_config: list) -> float:
        """余弦退火衰减 lambda。"""
        total = getattr(self.trainer, "max_epochs", None) or 1
        progress = float(self.current_epoch) / float(max(total - 1, 1))
        start, end = float(range_config[0]), float(range_config[1])
        return self._cosine_decay(progress, start, end)

    # ------------------------------------------------------------------
    #  损失校准
    # ------------------------------------------------------------------

    def _calibrate_loss(
        self,
        task_loss: torch.Tensor,
        l_out: Optional[torch.Tensor],
        l_band: Optional[torch.Tensor],
    ) -> None:
        """首 batch 自动校准：使蒸馏损失与任务损失同量级。"""
        if self._calibrated:
            return
        with torch.no_grad():
            task_val = task_loss.abs().item()
            if l_out is not None and l_out.abs().item() > 1e-12:
                self._calib_scale_out = task_val / l_out.abs().item()
                self._calib_scale_out = max(0.01, min(100.0, self._calib_scale_out))
            if l_band is not None and l_band.abs().item() > 1e-12:
                self._calib_scale_band = task_val / l_band.abs().item()
                self._calib_scale_band = max(0.01, min(100.0, self._calib_scale_band))
        self._calibrated = True

    # ------------------------------------------------------------------
    #  训练步骤
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        """每 epoch 开始时：应用二值化阶段 + 蒸馏预热。"""
        super().on_train_epoch_start()
        if self.distillation_enabled:
            self._apply_distill_warmup()

    def training_step(self, batch, batch_nb):
        # 如果蒸馏未启用，走纯二值化路径
        if not self.distillation_enabled or self.teacher_model is None:
            result = super(BinaryAudioLightningModule, self).training_step(batch, batch_nb)
            if hasattr(self.audio_model, "clamp_all_binary_weights"):
                self.audio_model.clamp_all_binary_weights()
            return result

        # ---- 蒸馏路径 ----
        mixtures, targets, _ = batch

        # SpeedAug
        if self.config["training"].get("SpeedAug", False):
            mixtures, targets = self._apply_speed_aug(mixtures, targets)

        est_sources = self(mixtures)
        task_loss = self.loss_func["train"](est_sources, targets)
        self.log("train/task_loss", task_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, logger=True)

        # 教师前向
        with torch.no_grad():
            teacher_sources = self.teacher_model(mixtures)

        # 计算蒸馏损失
        l_out = None
        l_band = None
        if self.kd_type == "d3" and self.kd_loss_combined is not None:
            # D3: 使用 Combined_KDLoss 共享 PIT 排列
            l_out, l_band = self.kd_loss_combined(est_sources, teacher_sources)
        else:
            if self.kd_loss_out is not None:
                l_out = self.kd_loss_out(est_sources, teacher_sources)
            if self.kd_loss_band is not None:
                l_band = self.kd_loss_band(est_sources, teacher_sources)

        # 首 batch 校准
        if self.loss_calibration_enabled:
            self._calibrate_loss(task_loss, l_out, l_band)

        # lambda 调度
        lambda_out = self._resolve_lambda(self.lambda_out_range) if l_out is not None else 0.0
        lambda_band = self._resolve_lambda(self.lambda_band_range) if l_band is not None else 0.0

        # 总损失
        loss = task_loss
        if l_out is not None:
            loss = loss + self._calib_scale_out * lambda_out * l_out
            self.log("train/l_out", l_out, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, logger=True)
            self.log("train/lambda_out", lambda_out, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, logger=True)
        if l_band is not None:
            loss = loss + self._calib_scale_band * lambda_band * l_band
            self.log("train/l_band", l_band, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, logger=True)
            self.log("train/lambda_band", lambda_band, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, logger=True)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)

        # 二值权重 clamp
        if hasattr(self.audio_model, "clamp_all_binary_weights"):
            self.audio_model.clamp_all_binary_weights()

        return {"loss": loss}

    def _apply_speed_aug(self, mixtures, targets):
        """SpeedAug 数据增强。"""
        new_targets = []
        min_len = -1
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
                targets.shape[0], targets.shape[1], min_len,
                device=targets.device, dtype=torch.float,
            )
            for i, new_target in enumerate(new_targets):
                targets[:, i, :] = new_targets[i][:, 0:min_len]
            mixtures = targets.sum(1)
        return mixtures, targets
