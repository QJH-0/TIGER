import argparse

import torch
import yaml
from torch.profiler import ProfilerActivity, profile

import look2hear.models
from look2hear.layers.binary_layers import BinaryConv1d, BinaryLinear

try:
    from ptflops import get_model_complexity_info
except ImportError:  # pragma: no cover - optional dependency
    get_model_complexity_info = None


def build_model(config_path: str):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model = getattr(look2hear.models, config["audionet"]["audionet_name"])(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )
    return model, config


def count_binary_ops(model: torch.nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, BinaryConv1d):
            kernel = module.kernel_size[0]
            total += (
                module.out_channels
                * (module.in_channels // module.groups)
                * kernel
            )
        elif isinstance(module, BinaryLinear):
            total += module.in_features * module.out_features
    return total


def summarize_efficiency(model: torch.nn.Module, input_length: int) -> dict[str, float]:
    params = sum(parameter.numel() for parameter in model.parameters())
    fp32_flops = estimate_flops(model, input_length)
    binary_bops = count_binary_ops(model)
    equivalent_flops = fp32_flops - binary_bops + binary_bops / 64.0
    model_size_mb = params * 4 / (1024**2)
    return {
        "fp32_flops": float(fp32_flops),
        "binary_bops": float(binary_bops),
        "equivalent_flops": float(equivalent_flops),
        "params": float(params),
        "model_size_mb": float(model_size_mb),
    }


def estimate_flops(model: torch.nn.Module, input_length: int) -> float:
    if get_model_complexity_info is not None:
        flops, _ = get_model_complexity_info(
            model,
            (input_length,),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        return float(flops)

    example = torch.randn(1, input_length)
    model = model.cpu().eval()
    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as profiler:
        with torch.no_grad():
            model(example)
    return float(sum(event.flops for event in profiler.key_averages()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/tiger-small-kaggle-t4x2.yml",
        help="YAML config path used to instantiate the model.",
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=16000,
        help="Input waveform length for FLOPs estimation.",
    )
    args = parser.parse_args()

    model, _ = build_model(args.config)
    stats = summarize_efficiency(model, args.input_length)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
