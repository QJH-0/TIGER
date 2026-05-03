import os
import sys
import torch
from torch import Tensor
import argparse
import json
import yaml
import look2hear.datas
import look2hear.models
import look2hear.system
import look2hear.losses
import look2hear.metrics
import look2hear.utils
from look2hear.system import make_optimizer
from look2hear.models.teacher_tiger import load_teacher_tiger
from dataclasses import dataclass
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import *
from rich.console import Console
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from rich import print, reconfigure
from collections.abc import MutableMapping
from look2hear.utils import print_only, MyRichProgressBar, RichProgressBarTheme

import warnings

warnings.filterwarnings("ignore")

import wandb

#手动登录

"""
TIGER 训练脚本。

职责：
1. 根据 yaml 配置实例化 `datamodule`、`audio model`、优化器与（可选）学习率调度器。
2. 构建 Look2Hear 的 `System`（封装 loss / train / val / test 逻辑），并配置 Lightning Trainer。
3. 训练完成后保存：
   - `best_k_models.json`（checkpoint 中 top-k 模型的分数）
   - `best_model.pth`（将 best checkpoint 的 state_dict 加载到音频模型并序列化）
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="local/conf.yml",
    help="Full path to save best validation model",
)
parser.add_argument(
    "--epoch",
    type=int,
    default=None,
    help="Override training epochs (maps to training.epochs / Trainer max_epochs)",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=None,
    help="Override datamodule.data_config.batch_size (per GPU/rank).",
)

parser.add_argument(
    "--segment",
    type=float,
    default=None,
    help="Override datamodule.data_config.segment in seconds (e.g. 3.0 -> 48000 @16kHz).",
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume training from the current experiment's last.ckpt.",
)
parser.add_argument(
    "--resume_ckpt",
    type=str,
    default=None,
    help="Resume training from a specific checkpoint path, or use 'last'.",
)


def flatten_config(config, parent_key="", sep="_"):
    items = []
    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_config(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def sanitize_wandb_config(config):
    sanitized = {}
    for key, value in flatten_config(config).items():
        if value is None:
            sanitized[key] = "None"
        elif isinstance(value, Tensor):
            sanitized[key] = value.tolist()
        else:
            sanitized[key] = value
    return sanitized


def _slugify_wandb_name(value):
    text = str(value).strip().replace("_", "-").replace(" ", "-")
    return "".join(ch.lower() for ch in text if ch.isalnum() or ch in "-.")


def _compact_data_name_for_wandb(data_name):
    text = str(data_name).strip()
    for suffix in ("ModuleRemix", "DataModule", "Module"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text or str(data_name)


def _format_segment_for_wandb(segment):
    if segment == "na":
        return segment
    try:
        f = float(segment)
    except (TypeError, ValueError):
        return str(segment)
    if f == int(f):
        return str(int(f))
    return str(f)


def build_wandb_project_name(config):
    return "tiger-speech-separation-model"


def ensure_wandb_login(wandb_module=wandb):
    wandb_module.login()


def build_wandb_run_name(config):
    audionet_name = config.get("audionet", {}).get("audionet_name", "model")
    datamodule = config.get("datamodule", {})
    data_name = datamodule.get("data_name", "dataset")
    data_config = datamodule.get("data_config", {})
    batch_size = data_config.get("batch_size", "na")
    segment = data_config.get("segment", "na")
    exp_name = config.get("exp", {}).get("exp_name")

    parts = [
        _slugify_wandb_name(audionet_name),
        _slugify_wandb_name(_compact_data_name_for_wandb(data_name)),
        f"bs{batch_size}",
        f"seg{_format_segment_for_wandb(segment)}",
    ]
    if exp_name:
        parts.append(_slugify_wandb_name(exp_name))
    return "-".join(parts)


def save_wandb_run_metadata(wandb_logger, exp_dir):
    experiment = wandb_logger.experiment

    def _resolve_experiment_field(field_name):
        value = getattr(experiment, field_name, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                return None
        return value

    metadata = {
        "entity": _resolve_experiment_field("entity"),
        "project": _resolve_experiment_field("project"),
        "run_id": _resolve_experiment_field("id"),
        "run_name": _resolve_experiment_field("name"),
        "url": _resolve_experiment_field("url"),
    }
    with open(os.path.join(exp_dir, "wandb_run.json"), "w", encoding="utf-8") as outfile:
        json.dump(metadata, outfile, indent=2)


def configure_wandb_epoch_metrics(wandb_logger):
    # 统一把核心训练指标绑定到 epoch，并隐藏自动生成工作区里无关的辅助横轴。
    experiment = wandb_logger.experiment
    experiment.define_metric("epoch", hidden=True)
    experiment.define_metric("trainer/global_step", hidden=True)
    experiment.define_metric("train/loss", step_metric="epoch")
    experiment.define_metric("train/learning_rate", step_metric="epoch")
    experiment.define_metric("val/loss", step_metric="epoch")
    experiment.define_metric("val/si_snr", step_metric="epoch")


def sanitize_exp_name_for_path(exp_name) -> str:
    """将 yaml 中的 exp_name 转为安全的目录名，避免不同实验写到同一路径或非法路径。"""
    if exp_name is None:
        return "unnamed-exp"
    s = str(exp_name).strip()
    if not s:
        return "unnamed-exp"
    for ch in '<>:"/\\|?*':
        s = s.replace(ch, "_")
    s = s.rstrip(". ")
    return s or "unnamed-exp"


def lightning_checkpoint_dir(exp_dir: str) -> str:
    """Lightning ModelCheckpoint 写入目录：与 conf / best_model.pth 隔离，减少误加载。"""
    return os.path.join(exp_dir, "checkpoints")


def build_checkpoint_callback(checkpoint_dir):
    # 训练中只保留两个 checkpoint：last.ckpt（续训）和最优 best（验证集最小 val/loss）。
    # enable_version_counter=False：续训/重复保存时覆盖同名文件，避免 best-vN.ckpt、last-vN.ckpt 堆积。
    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        verbose=True,
        save_last=True,
        enable_version_counter=False,
    )


def resolve_trainer_runtime_config(config, cuda_available=None):
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()

    if cuda_available:
        devices = config["training"]["gpus"]
        accelerator = "cuda"
        use_ddp = isinstance(devices, (list, tuple)) and len(devices) > 1
        sync_batchnorm = use_ddp
        precision = config["training"].get("precision", "16-mixed")
    else:
        devices = 1
        accelerator = "cpu"
        use_ddp = False
        sync_batchnorm = False
        precision = "32-true"

    return {
        "devices": devices,
        "accelerator": accelerator,
        "use_ddp": use_ddp,
        "sync_batchnorm": sync_batchnorm,
        "precision": precision,
    }


def resolve_datamodule_runtime_config(config, cuda_available=None):
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()

    data_config = dict(config["datamodule"]["data_config"])
    if not cuda_available:
        data_config["num_workers"] = 0
        data_config["pin_memory"] = False
        data_config["persistent_workers"] = False
    return data_config


def _resolve_default_last_ckpt_path(exp_dir):
    """优先新目录 checkpoints/last.ckpt，兼容旧版直接写在实验根目录下的 last.ckpt。"""
    preferred = os.path.join(lightning_checkpoint_dir(exp_dir), "last.ckpt")
    legacy = os.path.join(exp_dir, "last.ckpt")
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(legacy):
        return legacy
    return preferred


def resolve_resume_checkpoint_path(config, exp_dir):
    # 断点续训只认 Lightning checkpoint，不从 best_model.pth 恢复训练状态。
    main_args = config.get("main_args", {})
    resume_setting = main_args.get("resume_from_checkpoint")
    if not resume_setting:
        return None

    if isinstance(resume_setting, str):
        # 支持显式写 "last" 或直接给某个 .ckpt 的绝对/相对路径。
        if resume_setting.lower() == "last":
            return _resolve_default_last_ckpt_path(exp_dir)
        return resume_setting

    if resume_setting is True:
        # 简写为 true 时，默认恢复当前实验目录下的 last.ckpt。
        return _resolve_default_last_ckpt_path(exp_dir)

    return None


def resolve_cli_resume_override(plain_args):
    # CLI 优先级高于配置：显式给出 checkpoint 时直接使用，否则 --resume 恢复到 last.ckpt。
    resume_ckpt = getattr(plain_args, "resume_ckpt", None)
    if resume_ckpt:
        return resume_ckpt

    if getattr(plain_args, "resume", False):
        return True

    return None


def _checkpoint_int_scalar(value):
    if value is None:
        return None
    return int(value.item()) if hasattr(value, "item") else int(value)


def extract_resume_progress(checkpoint):
    """解析 Lightning checkpoint 中的进度（与 PyTorch Lightning 写入语义一致）。"""
    epoch_raw = checkpoint.get("epoch")
    global_step = checkpoint.get("global_step")

    if epoch_raw is None:
        return None

    lightning_completed = _checkpoint_int_scalar(epoch_raw)
    progress = {
        # 与 trainer.current_epoch / checkpoint['epoch'] 一致：已完整跑完的训练 epoch 个数
        "lightning_completed_epochs": lightning_completed,
        # 人类习惯的「下一轮」序号（从 1 开始计数）
        "human_next_epoch": lightning_completed + 1,
        "global_step": _checkpoint_int_scalar(global_step),
    }
    return progress


def build_resume_summary_message(resume_ckpt_path, checkpoint, max_epochs=None):
    if not resume_ckpt_path:
        return None

    progress = extract_resume_progress(checkpoint)
    if progress is None:
        return f"【断点接力】{resume_ckpt_path}：检查点中缺少 epoch 元数据。"

    lc = progress["lightning_completed_epochs"]
    hn = progress["human_next_epoch"]
    gs = progress["global_step"]
    gs_txt = f"{gs}" if gs is not None else "?"

    if max_epochs is not None:
        if hn > max_epochs:
            return (
                f"【断点接力】{resume_ckpt_path}\n"
                f"  检查点 epoch 计数={lc}，已超过训练计划 {max_epochs} 轮；本次 fit 可能立即结束。"
                f" global_step={gs_txt}"
            )
        return (
            f"【断点接力】{resume_ckpt_path}\n"
            f"  检查点内 Lightning epoch={lc}（已完整结束 {lc} 个训练 epoch）；"
            f"续训进度条将从 Epoch {hn}/{max_epochs} 起接力至 {max_epochs}/{max_epochs}。"
            f" global_step={gs_txt}"
        )

    return (
        f"【断点接力】{resume_ckpt_path}：checkpoint epoch={lc}，下一显示轮次={hn}，global_step={gs_txt}"
    )


def log_resume_checkpoint_status(resume_ckpt_path, max_epochs=None):
    if not resume_ckpt_path:
        return

    checkpoint = torch.load(resume_ckpt_path, map_location="cpu")
    message = build_resume_summary_message(
        resume_ckpt_path, checkpoint, max_epochs=max_epochs
    )
    if message:
        print_only(message)


def resolve_export_checkpoint_path(checkpoint_callback, exp_dir):
    best_model_path = getattr(checkpoint_callback, "best_model_path", "") or ""
    if best_model_path:
        return best_model_path

    last_model_path = getattr(checkpoint_callback, "last_model_path", "") or ""
    if last_model_path and os.path.exists(last_model_path):
        return last_model_path

    preferred_last = os.path.join(lightning_checkpoint_dir(exp_dir), "last.ckpt")
    legacy_last = os.path.join(exp_dir, "last.ckpt")
    if os.path.exists(preferred_last):
        return preferred_last
    if os.path.exists(legacy_last):
        return legacy_last

    raise FileNotFoundError(
        "No checkpoint available for exporting best_model.pth. "
        "Neither best_model_path nor checkpoints/last.ckpt (nor legacy last.ckpt) exists."
    )


def extract_model_state_dict(checkpoint_payload):
    if not isinstance(checkpoint_payload, dict):
        raise TypeError("Unsupported checkpoint payload type.")

    if "state_dict" in checkpoint_payload:
        state_dict = checkpoint_payload["state_dict"]
    else:
        state_dict = checkpoint_payload

    if any(key.startswith("audio_model.") for key in state_dict.keys()):
        return {
            key[len("audio_model.") :]: value
            for key, value in state_dict.items()
            if key.startswith("audio_model.")
        }
    return state_dict


def normalize_checkpoint_keys_for_model(model, state_dict):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict

    model_state_keys = tuple(model.state_dict().keys())
    if not model_state_keys:
        return state_dict

    if any(key.startswith("model.") for key in state_dict.keys()):
        return state_dict

    if not any(key.startswith("model.") for key in model_state_keys):
        return state_dict

    prefixed_state_dict = {f"model.{key}": value for key, value in state_dict.items()}
    prefixed_keys = set(prefixed_state_dict.keys())
    model_keys = set(model_state_keys)
    original_overlap = len(set(state_dict.keys()) & model_keys)
    prefixed_overlap = len(prefixed_keys & model_keys)
    if prefixed_overlap > original_overlap:
        return prefixed_state_dict
    return state_dict


def summarize_checkpoint_load(model, state_dict, missing_keys, unexpected_keys):
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    missing_keys = list(missing_keys)
    unexpected_keys = list(unexpected_keys)
    matched_keys = sorted((model_keys & checkpoint_keys) - set(missing_keys))
    model_key_count = len(model_keys)
    matched_key_count = len(matched_keys)
    return {
        "model_key_count": model_key_count,
        "checkpoint_key_count": len(checkpoint_keys),
        "matched_key_count": matched_key_count,
        "missing_key_count": len(missing_keys),
        "unexpected_key_count": len(unexpected_keys),
        "matched_key_ratio": (matched_key_count / model_key_count) if model_key_count else 0.0,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "matched_keys_preview": matched_keys[:10],
    }


def format_checkpoint_load_summary(summary, init_label):
    missing_preview = ", ".join(summary["missing_keys"][:5]) if summary["missing_keys"] else "none"
    unexpected_preview = ", ".join(summary["unexpected_keys"][:5]) if summary["unexpected_keys"] else "none"
    return (
        f"[CheckpointLoad:{init_label}] path={summary['checkpoint_path']}\n"
        f"  matched={summary['matched_key_count']}/{summary['model_key_count']} "
        f"({summary['matched_key_ratio']:.4f}) | "
        f"missing={summary['missing_key_count']} | unexpected={summary['unexpected_key_count']}\n"
        f"  missing_preview={missing_preview}\n"
        f"  unexpected_preview={unexpected_preview}"
    )


def enforce_checkpoint_load_policy(summary, init_label, min_match_ratio=0.5):
    matched_key_count = summary["matched_key_count"]
    matched_key_ratio = summary["matched_key_ratio"]
    if matched_key_count <= 0 or matched_key_ratio < float(min_match_ratio):
        raise RuntimeError(
            f"{init_label} checkpoint load validation failed for {summary['checkpoint_path']}: "
            f"matched {matched_key_count}/{summary['model_key_count']} keys "
            f"({matched_key_ratio:.4f}) < required {float(min_match_ratio):.4f}. "
            f"missing={summary['missing_key_count']}, unexpected={summary['unexpected_key_count']}. "
            f"missing_preview={summary['missing_keys'][:5]} unexpected_preview={summary['unexpected_keys'][:5]}"
        )


def load_model_checkpoint(model, checkpoint_path, init_label="checkpoint", min_match_ratio=0.5):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = extract_model_state_dict(checkpoint)
    state_dict = normalize_checkpoint_keys_for_model(model, state_dict)
    # BinaryTIGER 在 converter 中插入了 RSign，导致 Sequential 索引偏移。
    # 加载旧 checkpoint 时需要重映射键名以匹配新结构。
    if hasattr(model, "remap_checkpoint_keys"):
        state_dict = model.remap_checkpoint_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    summary = summarize_checkpoint_load(model, state_dict, missing, unexpected)
    summary["checkpoint_path"] = checkpoint_path
    summary["init_label"] = init_label
    print_only(format_checkpoint_load_summary(summary, init_label))
    enforce_checkpoint_load_policy(summary, init_label=init_label, min_match_ratio=min_match_ratio)
    return summary


def apply_cli_overrides(arg_dic, plain_args):
    # 如果提供了 CLI epoch，优先覆盖配置中的训练轮数。
    if getattr(plain_args, "epoch", None) is not None:
        if "training" in arg_dic:
            arg_dic["training"]["epochs"] = plain_args.epoch

    # CLI overrides for datamodule.*
    # NOTE: Our YAML config is multi-level (datamodule -> data_config -> {batch_size, segment}),
    # so we explicitly wire these overrides instead of relying on prepare_parser_from_dict.
    if getattr(plain_args, "batch_size", None) is not None:
        if "datamodule" in arg_dic and "data_config" in arg_dic["datamodule"]:
            arg_dic["datamodule"]["data_config"]["batch_size"] = plain_args.batch_size
    if getattr(plain_args, "segment", None) is not None:
        if "datamodule" in arg_dic and "data_config" in arg_dic["datamodule"]:
            arg_dic["datamodule"]["data_config"]["segment"] = plain_args.segment

    # CLI 覆盖优化器学习率，便于快速做学习率扫描实验。
    if getattr(plain_args, "lr", None) is not None:
        if "optimizer" in arg_dic:
            arg_dic["optimizer"]["lr"] = plain_args.lr

    cli_resume_override = resolve_cli_resume_override(plain_args)
    if cli_resume_override is not None:
        arg_dic.setdefault("main_args", {})
        arg_dic["main_args"]["resume_from_checkpoint"] = cli_resume_override

    return arg_dic

def main(config):
    """
    使用 PyTorch Lightning 进行训练。

    参数
    - config (dict): 从 yaml 解析得到的配置字典，至少包含：
      - `datamodule.data_name` / `datamodule.data_config`
      - `audionet.audionet_name` / `audionet.audionet_config`
      - `optimizer`、可选的 `scheduler`
      - `loss.train` 与 `loss.val`
      - `training`（epochs、gpus、system、early_stop 等）
      - `exp.exp_name` / `main_args`（用于保存实验目录）
    返回
    - None
    """
    print_only(
        "Instantiating datamodule <{}>".format(config["datamodule"]["data_name"])
    )
    config["datamodule"]["data_config"] = resolve_datamodule_runtime_config(config)
    datamodule: object = getattr(look2hear.datas, config["datamodule"]["data_name"])(
        **config["datamodule"]["data_config"]
    )
    datamodule.setup()

    # Lightning DataLoader：用于训练/验证/测试阶段。
    train_loader, val_loader, test_loader = datamodule.make_loader
    
    # 构建音频模型与优化器
    print_only(
        "Instantiating AudioNet <{}>".format(config["audionet"]["audionet_name"])
    )
    model = getattr(look2hear.models, config["audionet"]["audionet_name"])(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )
    distill_config = config.get("distillation", {})
    student_init_ckpt = distill_config.get("student_init_ckpt")
    if student_init_ckpt:
        print_only(f"Loading student init checkpoint <{student_init_ckpt}>")
        load_model_checkpoint(model, student_init_ckpt, init_label="student_init")

    # 阶段一共享预热 checkpoint：从预训练的 RSign/RPReLU checkpoint 出发
    training_config = config.get("training", {})
    warmup_ckpt = training_config.get("warmup_ckpt")
    if warmup_ckpt:
        print_only(f"Loading warmup checkpoint <{warmup_ckpt}>")
        load_model_checkpoint(model, warmup_ckpt, init_label="warmup")

    # 初始化 EMA Scale（BinaryTIGER 专用，加载预训练权重后调用一次）
    if hasattr(model, "init_all_scales"):
        print_only("Initializing EMA scales from pretrained weights")
        model.init_all_scales()
    # import pdb; pdb.set_trace()
    print_only("Instantiating Optimizer <{}>".format(config["optimizer"]["optim_name"]))
    # param_groups 是蒸馏参数分组配置，不属于优化器构造参数，需要过滤掉
    optimizer_kwargs = {k: v for k, v in config["optimizer"].items() if k != "param_groups"}
    optimizer = make_optimizer(model.parameters(), **optimizer_kwargs)

    # 学习率调度器：根据配置选择开启或不启用。
    scheduler = None
    if config["scheduler"]["sche_name"]:
        print_only(
            "Instantiating Scheduler <{}>".format(config["scheduler"]["sche_name"])
        )
        if config["scheduler"]["sche_name"] != "DPTNetScheduler":
            scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"]["sche_name"])(
                optimizer=optimizer, **config["scheduler"]["sche_config"]
            )
        else:
            scheduler = {
                # DPTNetScheduler 采用 Lightning 的字典形式并指定 interval。
                "scheduler": getattr(look2hear.system.schedulers, config["scheduler"]["sche_name"])(
                    optimizer, len(train_loader) // config["datamodule"]["data_config"]["batch_size"], 64
                ),
                "interval": "step",
            }

    # Just after instantiating, save the args. Easy loading in the future.
    # 将当前配置落盘到实验目录，便于复现实验与定位参数来源。
    # 实验根目录名由 exp_name 派生（清洗非法字符），与 yaml 中 exp_name 一一对应，避免多配置写到同一路径。
    exp_folder = sanitize_exp_name_for_path(config["exp"]["exp_name"])
    config["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "Experiments", exp_folder
    )
    exp_dir = config["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(lightning_checkpoint_dir(exp_dir), exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(config, outfile)

    # 定义 Loss：分别构建 train/val 两套 loss 组件（sdr_type 可能不同）。
    print_only(
        "Instantiating Loss, Train <{}>, Val <{}>".format(
            config["loss"]["train"]["sdr_type"], config["loss"]["val"]["sdr_type"]
        )
    )
    loss_func = {
        "train": getattr(look2hear.losses, config["loss"]["train"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["train"]["sdr_type"]),
            **config["loss"]["train"]["config"],
        ),
        "val": getattr(look2hear.losses, config["loss"]["val"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["val"]["sdr_type"]),
            **config["loss"]["val"]["config"],
        ),
    }

    print_only("Instantiating System <{}>".format(config["training"]["system"]))
    # System 负责把 model + loss + optimizer + dataloaders 串起来，并提供 Lightning 的训练步骤。
    system = getattr(look2hear.system, config["training"]["system"])(
        audio_model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        config=config,
    )
    if config["training"]["system"] == "BinaryDistillAudioLitModule":
        teacher_ckpt = distill_config.get("teacher_ckpt")
        if not teacher_ckpt:
            raise ValueError("BinaryDistillAudioLitModule requires distillation.teacher_ckpt")
        teacher_model_kwargs = dict(config["audionet"]["audionet_config"])
        teacher_model_kwargs.pop("binary_config", None)
        print_only(f"Loading teacher checkpoint <{teacher_ckpt}>")
        system.teacher_model = load_teacher_tiger(
            checkpoint_path=teacher_ckpt,
            model_kwargs=teacher_model_kwargs,
            sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        )

    # Define callbacks
    print_only("Instantiating ModelCheckpoint")
    callbacks = []
    checkpoint_dir = lightning_checkpoint_dir(exp_dir)
    # 根据 val/loss_epoch 监控最优模型，并保留 top-k。
    checkpoint = build_checkpoint_callback(checkpoint_dir)
    callbacks.append(checkpoint)

    if config["training"]["early_stop"]:
        print_only("Instantiating EarlyStopping")
        callbacks.append(EarlyStopping(**config["training"]["early_stop"]))
    callbacks.append(MyRichProgressBar(theme=RichProgressBarTheme(), force_single_line=True))

    # Don't ask GPU if they are not available.
    # 单卡/无 GPU：走 Lightning 默认策略；多卡且明确选择多张 CUDA 卡：启用 DDP。
    gpus = config["training"]["gpus"] if torch.cuda.is_available() else None
    distributed_backend = "cuda" if torch.cuda.is_available() else None
    # 仅在明确选择了多张 CUDA 卡时启用 DDP。
    use_ddp = isinstance(gpus, (list, tuple)) and len(gpus) > 1

    trainer_runtime = resolve_trainer_runtime_config(config)

    # default logger used by trainer
    logger_dir = os.path.join(os.getcwd(), "Experiments", "tensorboard_logs")
    log_subdir = sanitize_exp_name_for_path(config["exp"]["exp_name"])
    os.makedirs(os.path.join(logger_dir, log_subdir), exist_ok=True)
    # comet_logger = TensorBoardLogger(logger_dir, name=config["exp"]["exp_name"])
    ensure_wandb_login()
    comet_logger = WandbLogger(
            name=build_wandb_run_name(config), 
            save_dir=os.path.join(logger_dir, log_subdir), 
            project=build_wandb_project_name(config),
            config=sanitize_wandb_config(config),
            # offline=True
    )
    configure_wandb_epoch_metrics(comet_logger)
    save_wandb_run_metadata(comet_logger, exp_dir)

    # 单卡走默认策略，多卡保留 DDP，避免单卡场景策略配置报错。
    # limit_train_batches=1.0 用于完整训练；若想快速调试可以改成更小比例。
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=trainer_runtime["devices"],
        accelerator=trainer_runtime["accelerator"],
        strategy=DDPStrategy(find_unused_parameters=True) if trainer_runtime["use_ddp"] else "auto",
        accumulate_grad_batches=config["training"].get("accumulate_grad_batches", 1),
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        logger=comet_logger,
        sync_batchnorm=trainer_runtime["sync_batchnorm"],
        # 训练精度：用配置 training.precision 控制（默认 16-mixed 省显存），避免依赖 Lightning CLI 透传参数格式。
        precision=trainer_runtime["precision"],
        # num_sanity_val_steps=0,
        # sync_batchnorm=True,
        # fast_dev_run=True,
    )
    resume_ckpt_path = resolve_resume_checkpoint_path(config, exp_dir)
    planned_epochs = config["training"]["epochs"]
    if resume_ckpt_path:
        log_resume_checkpoint_status(resume_ckpt_path, max_epochs=planned_epochs)
    else:
        print_only(
            f"【全新训练】进度条按 Epoch 1/{planned_epochs} … {planned_epochs}/{planned_epochs} 显示（共 {planned_epochs} 轮）。"
        )
    # 恢复训练时把 ckpt_path 交给 Trainer；非恢复场景传 None 即可。
    trainer.fit(system, ckpt_path=resume_ckpt_path)
    print_only("Finished Training")

    # 保存 best_k_models：用于后续查看 checkpoint 分数分布。
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    # 将 best checkpoint 的 state_dict 加载回 system，然后把模型移到 CPU 并序列化音频模型权重。
    export_ckpt_path = resolve_export_checkpoint_path(checkpoint, exp_dir)
    state_dict = torch.load(export_ckpt_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.audio_model.serialize()
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


def cleanup_training_runtime(
    trainer=None,
    wandb_module=wandb,
    cuda_module=torch.cuda,
    distributed_module=torch.distributed,
):
    if trainer is not None:
        trainer.teardown()
    if wandb_module is not None:
        wandb_module.finish()
    if cuda_module is not None:
        cuda_module.empty_cache()
    if distributed_module is not None and distributed_module.is_initialized():
        distributed_module.destroy_process_group()


if __name__ == "__main__":
    from pprint import pprint
    from look2hear.utils.parser_utils import (
        prepare_parser_from_dict,
        parse_args_as_dict,
    )

    args = parser.parse_args()
    # Windows 下默认编码可能是 GBK；配置文件通常是 UTF-8（有时带 BOM）。
    try:
        with open(args.conf_dir, "r", encoding="utf-8") as f:
            def_conf = yaml.safe_load(f)
    except UnicodeDecodeError:
        with open(args.conf_dir, "r", encoding="utf-8-sig") as f:
            def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)

    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    # pprint(arg_dic)
    arg_dic = apply_cli_overrides(arg_dic, plain_args)
    try:
        main(arg_dic)
    finally:
        cleanup_training_runtime()
    print("TRAINING_SCRIPT_DONE")
