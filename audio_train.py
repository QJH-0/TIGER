import os
import sys
import torch
from torch import Tensor
import argparse
import json
import look2hear.datas
import look2hear.models
import look2hear.system
import look2hear.losses
import look2hear.metrics
import look2hear.utils
from look2hear.system import make_optimizer
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
# wandb.login()

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


def build_wandb_project_name(config):
    return "tiger-speech-separation"


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
        _slugify_wandb_name(data_name),
        f"bs{batch_size}",
        f"seg{segment}",
    ]
    if exp_name:
        parts.append(_slugify_wandb_name(exp_name))
    return "-".join(parts)


def save_wandb_run_metadata(wandb_logger, exp_dir):
    experiment = wandb_logger.experiment
    metadata = {
        "entity": getattr(experiment, "entity", None),
        "project": getattr(experiment, "project", None),
        "run_id": getattr(experiment, "id", None),
        "run_name": getattr(experiment, "name", None),
        "url": getattr(experiment, "url", None),
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


def build_checkpoint_callback(checkpoint_dir):
    # 训练中只保留两个 checkpoint：last.ckpt（续训）和最优 best（验证集最小 val/loss）。
    return ModelCheckpoint(
        checkpoint_dir,
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        verbose=True,
        save_last=True,
    )


def resolve_resume_checkpoint_path(config, exp_dir):
    # 断点续训只认 Lightning checkpoint，不从 best_model.pth 恢复训练状态。
    main_args = config.get("main_args", {})
    resume_setting = main_args.get("resume_from_checkpoint")
    if not resume_setting:
        return None

    if isinstance(resume_setting, str):
        # 支持显式写 "last" 或直接给某个 .ckpt 的绝对/相对路径。
        if resume_setting.lower() == "last":
            return os.path.join(exp_dir, "last.ckpt")
        return resume_setting

    if resume_setting is True:
        # 简写为 true 时，默认恢复当前实验目录下的 last.ckpt。
        return os.path.join(exp_dir, "last.ckpt")

    return None


def resolve_export_checkpoint_path(checkpoint_callback, exp_dir):
    best_model_path = getattr(checkpoint_callback, "best_model_path", "") or ""
    if best_model_path:
        return best_model_path

    last_model_path = getattr(checkpoint_callback, "last_model_path", "") or ""
    if last_model_path and os.path.exists(last_model_path):
        return last_model_path

    fallback_last = os.path.join(exp_dir, "last.ckpt")
    if os.path.exists(fallback_last):
        return fallback_last

    raise FileNotFoundError(
        "No checkpoint available for exporting best_model.pth. "
        "Neither best_model_path nor last.ckpt exists."
    )

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
    # import pdb; pdb.set_trace()
    print_only("Instantiating Optimizer <{}>".format(config["optimizer"]["optim_name"]))
    optimizer = make_optimizer(model.parameters(), **config["optimizer"])

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
    config["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", config["exp"]["exp_name"]
    )
    exp_dir = config["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
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

    # Define callbacks
    print_only("Instantiating ModelCheckpoint")
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir)
    # 根据 val/loss_epoch 监控最优模型，并保留 top-k。
    checkpoint = build_checkpoint_callback(checkpoint_dir)
    callbacks.append(checkpoint)

    if config["training"]["early_stop"]:
        print_only("Instantiating EarlyStopping")
        callbacks.append(EarlyStopping(**config["training"]["early_stop"]))
    callbacks.append(MyRichProgressBar(theme=RichProgressBarTheme()))

    # Don't ask GPU if they are not available.
    # 单卡/无 GPU：走 Lightning 默认策略；多卡且明确选择多张 CUDA 卡：启用 DDP。
    gpus = config["training"]["gpus"] if torch.cuda.is_available() else None
    distributed_backend = "cuda" if torch.cuda.is_available() else None
    # 仅在明确选择了多张 CUDA 卡时启用 DDP。
    use_ddp = isinstance(gpus, (list, tuple)) and len(gpus) > 1

    # default logger used by trainer
    logger_dir = os.path.join(os.getcwd(), "Experiments", "tensorboard_logs")
    os.makedirs(os.path.join(logger_dir, config["exp"]["exp_name"]), exist_ok=True)
    # comet_logger = TensorBoardLogger(logger_dir, name=config["exp"]["exp_name"])
    comet_logger = WandbLogger(
            name=build_wandb_run_name(config), 
            save_dir=os.path.join(logger_dir, config["exp"]["exp_name"]), 
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
        devices=gpus,
        accelerator=distributed_backend,
        strategy=DDPStrategy(find_unused_parameters=True) if use_ddp else "auto",
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        logger=comet_logger,
        sync_batchnorm=True,
        precision="16-mixed",
        # num_sanity_val_steps=0,
        # sync_batchnorm=True,
        # fast_dev_run=True,
    )
    resume_ckpt_path = resolve_resume_checkpoint_path(config, exp_dir)
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


if __name__ == "__main__":
    import yaml
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
    # If provided, allow CLI to override training epochs from the yaml.
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

    main(arg_dic)
