"""二值化训练系统。

管理二值训练阶段切换（warmup -> binary -> finetune），
支持模块级冻结（阶段零敏感度验证）、检查点验证、history.csv / final_metrics.json 输出。

BinaryConv 的 EMA Scale 在每次 ``optimizer.step()`` 之后更新（见 ``optimizer_step``），
与梯度累积语义一致：仅在实际执行优化步时同步。
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint

from .audio_litmodule import AudioLightningModule
from ..layers.binary_layers import BinaryConv1d, BinaryConv2d, BinaryLinear, RSign, RPReLU, classify_binary_module


class BinaryAudioLightningModule(AudioLightningModule):
    """二值模型训练系统。

    TIGER 主体结构不变，这里只负责管理二值训练阶段切换。
    支持通过 training.freeze_scope 配置模块级冻结。
    支持通过 training.checkpoint_epoch/checkpoint_threshold 进行检查点验证。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        training_config = self.config.get("training", {})
        self.binary_stage_epochs = training_config.get("binary_stage_epochs", {})
        self._printed_binarization_summary = False

        # 模块级冻结配置（阶段零敏感度验证用）
        self.freeze_scope: list[str] = training_config.get("freeze_scope", [])

        # 消融实验开关
        ablation_config = training_config.get("ablation", {})
        self.ablation_disable_rsign: bool = bool(ablation_config.get("disable_rsign", False))
        self.ablation_skip_warmup: bool = bool(ablation_config.get("skip_warmup", False))
        self.ablation_use_original_prelu: bool = bool(ablation_config.get("use_original_prelu", False))

        # 检查点验证配置
        self.checkpoint_epoch: int = int(training_config.get("checkpoint_epoch", 0))
        self.checkpoint_threshold: float = float(training_config.get("checkpoint_threshold", 0.0))
        self.fp32_baseline_val_loss: float = float(training_config.get("fp32_baseline_val_loss", 0.0))

        # history.csv 输出
        self._history_csv_path: str | None = None
        self._history_csv_written_header: bool = False

        # 训练结束是否写入 final_metrics.json（《综合实验方案》§5.3 / §6.3）
        self._write_final_metrics: bool = bool(training_config.get("write_final_metrics", True))

    def _maybe_update_binary_ema_scales(self) -> None:
        """在 optimizer.step() 之后更新 BinaryConv 的 EMA Scale（《二值化技术方案》）。"""
        if not self.training:
            return
        if hasattr(self.audio_model, "update_all_ema_scales"):
            self.audio_model.update_all_ema_scales()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        """执行优化步后更新二值卷积的 EMA Scale（梯度累积时仅最后 micro-batch 会触发本方法）。"""
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure=optimizer_closure)
        self._maybe_update_binary_ema_scales()

    def _resolve_stage(self) -> str:
        """解析当前训练阶段。

        兼容两套配置：
        1. 旧版：warmup -> binary
        2. 新版：activation_warmup -> weight_binarize -> finetune
        """
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
        """Apply the paper-aligned binary-training stage settings."""
        stage = self._resolve_stage()

        if self.ablation_skip_warmup and stage in {"activation_warmup", "warmup"}:
            stage = "weight_binarize" if "weight_binarize" in self.binary_stage_epochs else "binary"

        if hasattr(self.audio_model, "set_binary_training"):
            self.audio_model.set_binary_training(stage in {"binary", "weight_binarize", "finetune"})

        for module in self.audio_model.modules():
            if isinstance(module, RPReLU):
                module.active = True

        if stage in {"binary", "weight_binarize", "finetune"} and hasattr(self.audio_model, "clamp_all_binary_weights"):
            self.audio_model.clamp_all_binary_weights()
        return stage

    def _apply_ablation(self) -> None:
        """应用消融实验开关。

        - disable_rsign: 禁用 RSign（激活二值化），使其变为直通
        - use_original_prelu: 用原始 PReLU 替代 RPReLU
        """
        if self.ablation_disable_rsign:
            for module in self.audio_model.modules():
                if isinstance(module, RSign):
                    module.disabled = True

        if self.ablation_use_original_prelu:
            for name, module in self.audio_model.named_modules():
                if isinstance(module, RPReLU):
                    prelu = _AmpCompatiblePReLU(num_parameters=1)
                    prelu.weight.data.fill_(1.0)
                    parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
                    parent = self.audio_model
                    for part in parent_name.split("."):
                        if part:
                            parent = getattr(parent, part)
                    setattr(parent, attr_name, prelu)

    def _apply_module_freeze(self) -> None:
        """Apply the paper-aligned freezing rules for each training stage."""
        stage = self._resolve_stage()

        if stage in {"activation_warmup", "warmup"}:
            for _, param in self.audio_model.named_parameters():
                param.requires_grad = False
            for module in self.audio_model.modules():
                if isinstance(module, (RSign, RPReLU)):
                    for param in module.parameters():
                        param.requires_grad = True
            return

        if not self.freeze_scope:
            return

        for _, param in self.audio_model.named_parameters():
            param.requires_grad = False

        for name, module in self.audio_model.named_modules():
            category = self._classify_module(name)
            if category in self.freeze_scope:
                for param in module.parameters():
                    param.requires_grad = True

        for module in self.audio_model.modules():
            if isinstance(module, (RSign, RPReLU)):
                for param in module.parameters():
                    param.requires_grad = True

    @staticmethod
    def _classify_module(full_name: str) -> str | None:
        """根据模块路径名返回模块类别（委托给共享函数）。"""
        return classify_binary_module(full_name)

    def _write_history_csv(self) -> None:
        """将当前 epoch 的训练指标追加写入 history.csv。"""
        if self._history_csv_path is None:
            return

        try:
            # 获取指标
            train_loss = self.trainer.callback_metrics.get("train/loss")
            val_loss = self.trainer.callback_metrics.get("val/loss")
            lr = self.trainer.optimizers[0].param_groups[0]["lr"] if self.trainer.optimizers else 0.0

            train_loss_val = train_loss.item() if train_loss is not None else 0.0
            val_loss_val = val_loss.item() if val_loss is not None else 0.0

            # 写入 CSV
            mode = "a" if os.path.exists(self._history_csv_path) else "w"
            with open(self._history_csv_path, mode, newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if mode == "w":
                    writer.writerow(["epoch", "train_loss", "val_loss", "lr"])
                writer.writerow([self.current_epoch, train_loss_val, val_loss_val, lr])
        except Exception:
            pass  # 静默失败，不影响训练

    @staticmethod
    def _scalar_metric(value: Any) -> float | None:
        """将 callback 中的标量张量转为 Python float。"""
        if value is None:
            return None
        if hasattr(value, "detach"):
            return float(value.detach().cpu().item())
        return float(value)

    def _build_final_metrics_payload(self) -> dict[str, Any]:
        """组装训练结束导出的 final_metrics.json 内容。"""
        training = self.config.get("training", {})
        exp = self.config.get("exp", {})
        main_args = self.config.get("main_args", {})

        payload: dict[str, Any] = {
            "config": {
                "exp_name": exp.get("exp_name"),
                "exp_dir": main_args.get("exp_dir"),
                "binary_stage_epochs": training.get("binary_stage_epochs"),
                "freeze_scope": training.get("freeze_scope", []),
                "total_epochs_completed": int(self.current_epoch) + 1,
                "max_epochs": int(self.trainer.max_epochs) if self.trainer and self.trainer.max_epochs else None,
            },
            "metrics": {},
            "sensitivity": {},
            "efficiency": {},
        }

        if self.trainer is not None:
            cbm = self.trainer.callback_metrics
            metric_keys = (
                "val/loss",
                "train/loss",
                "train/task_loss",
                "train/l_out",
                "train/l_band",
            )
            for key in metric_keys:
                if key in cbm:
                    v = self._scalar_metric(cbm[key])
                    if v is not None:
                        payload["metrics"][key.replace("/", "_")] = v

            best_score = None
            best_path = None
            for cb in self.trainer.callbacks:
                if isinstance(cb, ModelCheckpoint):
                    if cb.best_model_score is not None:
                        best_score = self._scalar_metric(cb.best_model_score)
                    if getattr(cb, "best_model_path", None):
                        best_path = str(cb.best_model_path)
                    break
            if best_score is not None:
                payload["metrics"]["best_val_loss"] = best_score
            if best_path:
                payload["metrics"]["best_checkpoint_path"] = best_path

        if hasattr(self.audio_model, "get_binarization_summary"):
            try:
                summary = self.audio_model.get_binarization_summary()
                payload["efficiency"] = {
                    k: summary[k]
                    for k in (
                        "binary_params",
                        "total_params",
                        "binary_ratio_percent",
                        "fp32_estimated_size_mb",
                        "packed_binary_size_mb",
                    )
                    if k in summary
                }
            except Exception:
                pass

        return payload

    def on_fit_end(self) -> None:
        """训练结束时写入 final_metrics.json（与 history.csv 同目录）。"""
        super().on_fit_end()
        if not self._write_final_metrics:
            return
        exp_dir = self.config.get("main_args", {}).get("exp_dir")
        if not exp_dir:
            return
        out_path = os.path.join(exp_dir, "final_metrics.json")
        try:
            payload = self._build_final_metrics_payload()
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            self.print(f"[Binary] 已写入 final_metrics.json: {out_path}")
        except Exception as exc:
            self.print(f"[Binary] 写入 final_metrics.json 失败: {exc}")

    def _check_checkpoint_validation(self) -> None:
        """检查点验证：在指定 epoch 对比 FP32 基线，超过阈值则提前终止。"""
        if self.checkpoint_epoch <= 0:
            return
        if self.current_epoch != self.checkpoint_epoch:
            return
        if self.fp32_baseline_val_loss <= 0:
            return

        val_loss = self.trainer.callback_metrics.get("val/loss")
        if val_loss is None:
            return

        val_loss_val = val_loss.item()
        increase_pct = (val_loss_val - self.fp32_baseline_val_loss) / abs(self.fp32_baseline_val_loss)

        self.print(
            f"[Checkpoint Validation] epoch={self.current_epoch} "
            f"val_loss={val_loss_val:.4f} baseline={self.fp32_baseline_val_loss:.4f} "
            f"increase={increase_pct*100:.1f}% threshold={self.checkpoint_threshold*100:.1f}%"
        )

        if increase_pct > self.checkpoint_threshold:
            self.print(
                f"[Checkpoint Validation] FAILED: increase {increase_pct*100:.1f}% > threshold {self.checkpoint_threshold*100:.1f}%, stopping training."
            )
            self.trainer.should_stop = True

    def on_train_epoch_start(self) -> None:
        self._apply_stage()
        self._apply_module_freeze()

    def on_fit_start(self) -> None:
        super().on_fit_start()

        # 应用消融实验开关（一次性）
        self._apply_ablation()

        # 初始化 history.csv 路径
        exp_dir = self.config.get("main_args", {}).get("exp_dir")
        if exp_dir:
            self._history_csv_path = os.path.join(exp_dir, "history.csv")

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

    def on_validation_epoch_end(self) -> None:
        """每 epoch 结束时：写入 history.csv + 检查点验证。"""
        super().on_validation_epoch_end()
        self._write_history_csv()
        self._check_checkpoint_validation()

    def training_step(self, batch, batch_nb):
        result = super().training_step(batch, batch_nb)
        if hasattr(self.audio_model, "clamp_all_binary_weights"):
            self.audio_model.clamp_all_binary_weights()
        return result


class _AmpCompatiblePReLU(nn.PReLU):
    """兼容 AMP 的 PReLU：前向时将权重对齐到输入 dtype/device。"""

    def forward(self, input):
        weight = self.weight
        if weight.device != input.device or weight.dtype != input.dtype:
            weight = weight.to(device=input.device, dtype=input.dtype)
        return F.prelu(input, weight)
