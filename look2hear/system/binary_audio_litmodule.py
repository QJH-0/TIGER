from __future__ import annotations

from .audio_litmodule import AudioLightningModule


class BinaryAudioLightningModule(AudioLightningModule):
    # 二值模型训练系统。
    # TIGER 主体结构不变，这里只负责管理二值训练阶段切换。
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        training_config = self.config.get("training", {})
        self.binary_stage_epochs = training_config.get("binary_stage_epochs", {})
        self._printed_binarization_summary = False

    def _resolve_stage(self) -> str:
        # 兼容两套配置：
        # 1. 旧版：warmup -> binary
        # 2. 新版：activation_warmup -> weight_binarize -> finetune
        if "activation_warmup" in self.binary_stage_epochs or "weight_binarize" in self.binary_stage_epochs:
            activation_warmup_epochs = int(self.binary_stage_epochs.get("activation_warmup", 0))
            weight_binarize_until = int(self.binary_stage_epochs.get("weight_binarize", activation_warmup_epochs))
            if self.current_epoch < activation_warmup_epochs:
                return "activation_warmup"
            if self.current_epoch < weight_binarize_until:
                return "weight_binarize"
            return "finetune"

        warmup_epochs = int(self.binary_stage_epochs.get("warmup", 0))
        if self.current_epoch < warmup_epochs:
            return "warmup"
        return "binary"

    def _apply_stage(self) -> str:
        stage = self._resolve_stage()
        if hasattr(self.audio_model, "set_binary_training"):
            self.audio_model.set_binary_training(stage in {"binary", "weight_binarize", "finetune"})
        if stage in {"binary", "weight_binarize", "finetune"} and hasattr(self.audio_model, "clamp_all_binary_weights"):
            self.audio_model.clamp_all_binary_weights()
        return stage

    def on_train_epoch_start(self) -> None:
        self._apply_stage()

    def on_fit_start(self) -> None:
        # 启动时打印哪些模块被二值化、哪些模块被保护，便于核对映射关系。
        if self._printed_binarization_summary:
            return
        if not hasattr(self.audio_model, "get_binarization_summary"):
            return

        summary = self.audio_model.get_binarization_summary()
        binary_modules = summary.get("binary_module_names", [])
        protected_modules = summary.get("protected_module_names", [])
        binary_preview = ", ".join(binary_modules[:10]) if binary_modules else "<none>"
        protected_preview = ", ".join(protected_modules[:10]) if protected_modules else "<none>"
        if len(binary_modules) > 10:
            binary_preview += f", ... (+{len(binary_modules) - 10} more)"
        if len(protected_modules) > 10:
            protected_preview += f", ... (+{len(protected_modules) - 10} more)"
        binary_ratio_percent = summary.get(
            "binary_ratio_percent",
            round(float(summary.get("binary_ratio", 0.0)) * 100.0, 2),
        )

        self.print(
            "[BinaryTIGER] binarized_modules="
            f"{len(binary_modules)} binary_params={summary['binary_params']}/{summary['total_params']} "
            f"({binary_ratio_percent}%)"
        )
        # 大小估计说明：
        # - fp32_estimated：与 Lightning Summary 口径一致（按 FP32 4 bytes/param 粗估）
        # - actual：按当前参数 dtype 的真实字节数
        # - packed_binary：推理端假设“二值层权重 1-bit 打包、bias/其余层 FP32”的等效大小
        if "fp32_estimated_size_mb" in summary:
            self.print(
                "[BinaryTIGER] size_estimates_mb="
                f"fp32_estimated={summary['fp32_estimated_size_mb']} "
                f"actual={summary.get('actual_param_size_mb', 'n/a')} "
                f"packed_binary={summary.get('packed_binary_size_mb', 'n/a')}"
            )
        self.print(f"[BinaryTIGER] binarized_parts: {binary_preview}")
        self.print(f"[BinaryTIGER] protected_parts: {protected_preview}")
        self._printed_binarization_summary = True

    def training_step(self, batch, batch_nb):
        stage = self._apply_stage()
        result = super().training_step(batch, batch_nb)
        if stage in {"binary", "weight_binarize", "finetune"} and hasattr(self.audio_model, "clamp_all_binary_weights"):
            self.audio_model.clamp_all_binary_weights()
        return result
