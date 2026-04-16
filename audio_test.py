import os
import random
from typing import Union
import soundfile as sf
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from scipy.io import wavfile
import warnings
import torchaudio
import wandb
warnings.filterwarnings("ignore")
import look2hear.models
import look2hear.datas
from look2hear.metrics import MetricsTracker
from look2hear.utils import tensors_to_device, RichProgressBarTheme, MyMetricsTextColumn, BatchesProcessedColumn
from audio_train import build_wandb_project_name

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

"""
TIGER 测试/评估脚本。

主要流程：
1. 根据配置加载训练好的 best 模型权重（`best_model.pth`）。
2. 构建数据模块并取测试集 `test_set`。
3. 对每个样本执行前向推理，计算并累计指标，最终写入 `metrics.csv`。
4. 可选：在指定样本索引（当前为 `idx==825`）时额外保存每个源的 wav 文件。
"""

parser = argparse.ArgumentParser()
parser.add_argument("--conf_dir",
                    default="local/mixit_conf.yml",
                    help="Full path to save best validation model")


compute_metrics = ["si_sdr", "sdr"]
# 指定要计算的指标名称（具体是否被 MetricsTracker 使用取决于其实现）。
os.environ['CUDA_VISIBLE_DEVICES'] = "8"
# 显式指定可见 GPU（用于固定推理使用的卡号；不影响 CPU 逻辑）。

def resolve_eval_model_source(config, exp_dir):
    # 评估默认读取当前实验导出的 best_model.pth；
    # 只有显式指定 test_model_path 时才切到预训练模型。
    main_args = config["train_conf"].setdefault("main_args", {})
    requested_model = main_args.get("test_model_path")
    if requested_model:
        return {"source_type": "pretrained", "path": requested_model}

    return {"source_type": "best", "path": os.path.join(exp_dir, "best_model.pth")}


def build_test_summary(metrics_row):
    return {
        "test/sdr": float(metrics_row["sdr"]),
        "test/sdr_i": float(metrics_row["sdr_i"]),
        "test/si_snr": float(metrics_row["si-snr"]),
        "test/si_snr_i": float(metrics_row["si-snr_i"]),
    }


def print_test_summary(summary, print_fn=print):
    print_fn("Test Summary")
    for key, value in summary.items():
        print_fn(f"{key}: {value:.3f}")


def load_wandb_run_metadata(exp_dir):
    metadata_path = os.path.join(exp_dir, "wandb_run.json")
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as infile:
        return json.load(infile)


def log_test_summary_to_wandb(summary, train_conf, exp_dir, wandb_module=wandb):
    metadata = load_wandb_run_metadata(exp_dir)
    if not metadata:
        raise FileNotFoundError(f"W&B run metadata not found in {exp_dir}")

    entity = metadata.get("entity")
    project = metadata.get("project") or build_wandb_project_name(train_conf)
    run_id = metadata.get("run_id")
    if not entity or not project or not run_id:
        raise ValueError("Incomplete W&B run metadata; expected entity, project, and run_id.")

    api = wandb_module.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    run.summary.update(summary)


def main(config):
    """
    执行测试并计算指标。

    参数
    - config (dict): 从 yaml 解析得到的配置字典，至少包含：
      - `train_conf.exp.exp_name`
      - `train_conf.audionet.audionet_name` 与 `train_conf.audionet.audionet_config`
      - `train_conf.datamodule.data_name` 与 `train_conf.datamodule.data_config`
      - `train_conf.training.gpus`
      - `train_conf.training` 下其它用于构建模型/数据的字段
    """
    metricscolumn = MyMetricsTextColumn(style=RichProgressBarTheme.metrics)
    # 原始进度条配置（使用 Unicode '•'，在 Windows GBK 控制台下会编码报错）：
    # progress = Progress(
    #     TextColumn("[bold blue]Testing", justify="right"),
    #     BarColumn(bar_width=None),
    #     "•",
    #     BatchesProcessedColumn(style=RichProgressBarTheme.batch_progress),
    #     "•",
    #     TransferSpeedColumn(),
    #     "•",
    #     TimeRemainingColumn(),
    #     "•",
    #     metricscolumn,
    # )
    # 修改：将分隔符改为 ASCII 字符，避免 GBK 编码错误。
    progress = Progress(
        TextColumn("[bold blue]Testing", justify="right"),
        BarColumn(bar_width=None),
        "-",
        BatchesProcessedColumn(style=RichProgressBarTheme.batch_progress),
        "-",
        TransferSpeedColumn(),
        "-",
        TimeRemainingColumn(),
        "-",
        metricscolumn,
    )

    train_conf = config["train_conf"]
    # 原始代码：假定配置里一定有 train_conf.main_args，当前 tiger-small.yml 中不存在会导致 KeyError。
    # config["train_conf"]["main_args"]["exp_dir"] = os.path.join(
    #     os.getcwd(), "Experiments", "checkpoint", config["train_conf"]["exp"]["exp_name"]
    # )
    # 修改：兼容无 main_args 的配置，自动补齐并计算 exp_dir。
    main_args = train_conf.setdefault("main_args", {})
    exp_name = train_conf["exp"]["exp_name"]
    exp_dir = os.path.join(os.getcwd(), "Experiments", "checkpoint", exp_name)
    main_args["exp_dir"] = exp_dir
    # 测试/实际分离不加载 .ckpt，而是加载 best_model.pth 或显式指定的预训练模型。
    model_source = resolve_eval_model_source(config, exp_dir)
    # import pdb; pdb.set_trace()
    # conf["train_conf"]["masknet"].update({"n_src": 2})

    # 原始模型加载方式（通过 HuggingFace Hub 的 from_pretrained，本地路径会被当成 repo_id，导致校验失败）：
    # model = getattr(look2hear.models, config["train_conf"]["audionet"]["audionet_name"]).from_pretrained(
    #     model_path,
    #     sample_rate=config["train_conf"]["datamodule"]["data_config"]["sample_rate"],
    #     **config["train_conf"]["audionet"]["audionet_config"],
    # )
    # 修改：本地实例化模型，再从 best_model.pth 加载 state_dict。
    model_cls = getattr(look2hear.models, train_conf["audionet"]["audionet_name"])
    if model_source["source_type"] == "pretrained":
        # 评估预训练模型时，直接走模型类自带的 from_pretrained。
        model = model_cls.from_pretrained(
            model_source["path"],
            sample_rate=train_conf["datamodule"]["data_config"]["sample_rate"],
            **train_conf["audionet"]["audionet_config"],
        )
    else:
        # 评估本地训练结果时，只读取导出的 best_model.pth 权重，不读取训练 checkpoint。
        model = model_cls(
            sample_rate=train_conf["datamodule"]["data_config"]["sample_rate"],
            **train_conf["audionet"]["audionet_config"],
        )
        pretrained = torch.load(model_source["path"], map_location="cpu", weights_only=False)
        if isinstance(pretrained, dict) and "state_dict" in pretrained:
            model.load_state_dict(pretrained["state_dict"], strict=True)
        else:
            model.load_state_dict(pretrained, strict=True)

    # 原始代码：只要配置里写了 gpus 就直接 .to("cuda")，在无 CUDA 环境会报错。
    # if config["train_conf"]["training"]["gpus"]:
    #     device = "cuda"
    #     model.to(device)
    # 修改：增加 torch.cuda.is_available() 判断，无 GPU 时自动在 CPU 上跑。
    if train_conf["training"]["gpus"] and torch.cuda.is_available():
        device = "cuda"
        model.to(device)
    model_device = next(model.parameters()).device

    datamodule: object = getattr(look2hear.datas, train_conf["datamodule"]["data_name"])(
        **train_conf["datamodule"]["data_config"]
    )
    datamodule.setup()
    # 原始版本使用 EchoSetDataModule 接口：make_sets
    # _, _ , test_set = datamodule.make_sets
    # 修改：优先走 make_sets，否则退回到 LightningDataModule 的 data_test（Libri2MixModuleRemix）。
    if hasattr(datamodule, "make_sets"):
        _, _, test_set = datamodule.make_sets
    else:
        test_set = datamodule.data_test
   
    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join(exp_dir, "results/")
    os.makedirs(ex_save_dir, exist_ok=True)
    metrics = MetricsTracker(
        save_file=os.path.join(ex_save_dir, "metrics.csv"))
    # 为性能进入 no_grad（避免 autograd 图构建；与下面 progress 循环配合使用）。
    torch.no_grad().__enter__()
    with progress: 
        for idx in progress.track(range(len(test_set))):
            # 原始代码将“metrics 计算 + 保存”都包在 `if idx == 825:` 内，导致除 idx==825 外不会累积指标。
            # 修改：对全测试集都进行前向并计算 metrics；仅在 idx==825 时保存 wav。
            # Forward the network on the mixture.
            mix, sources, key = tensors_to_device(
                test_set[idx], device=model_device
            )
            # 模型通常期望 shape 为 [B, T]（这里将单条样本扩展为 batch 维）。
            est_sources = model(mix[None])
            mix_np = mix
            sources_np = sources
            # 估计结果从 [B, n_src, T] 压回 [n_src, T]，方便 metrics 与保存逻辑使用。
            est_sources_np = est_sources.squeeze(0)

            # 原始（被注释）metrics 逻辑：
            # metrics(mix=mix_np,
            #         clean=sources_np,
            #         estimate=est_sources_np,
            #         key=key)
            # 修改：对每个 idx 都计算指标并写入 metrics.csv。
            metrics(
                mix=mix_np,
                clean=sources_np,
                estimate=est_sources_np,
                key=key,
            )

            if idx == 25:
                # 仅对特定 idx 保存分离结果的 wav，便于人工检查质量。
                save_dir = os.path.join("./result/TIGER", "idx{}".format(idx))
                # est_sources_np = normalize_tensor_wav(est_sources_np)
                for i in range(est_sources_np.shape[0]):
                    os.makedirs(
                        os.path.join(save_dir, "s{}/".format(i + 1)), exist_ok=True
                    )
                    # torchaudio.save(os.path.join(save_dir, "s{}/".format(i + 1)) + key, est_sources_np[i].unsqueeze(0).cpu(), 16000)
                    torchaudio.save(
                        os.path.join(save_dir, "s{}/".format(i + 1))
                        + key.split("/")[-1],
                        est_sources_np[i].unsqueeze(0).cpu(),
                        16000,
                    )
                # if idx % 50 == 0:
                #     metricscolumn.update(metrics.update())
    metrics_row = metrics.final()
    summary = build_test_summary(metrics_row)
    print_test_summary(summary)
    try:
        log_test_summary_to_wandb(summary, train_conf, exp_dir)
    except Exception as exc:
        print(f"W&B logging skipped: {exc}")


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    with open(args.conf_dir, "rb") as f:
        train_conf = yaml.safe_load(f)
    arg_dic["train_conf"] = train_conf
    # print(arg_dic)
    main(arg_dic)
