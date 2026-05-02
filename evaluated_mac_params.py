import argparse

import torch
import yaml
from torch.profiler import ProfilerActivity, profile

import look2hear.models
from look2hear.layers.binary_layers import BinaryConv1d, BinaryConv2d, BinaryLinear

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
        elif isinstance(module, BinaryConv2d):
            kh, kw = module.kernel_size
            total += (
                module.out_channels
                * (module.in_channels // module.groups)
                * kh
                * kw
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


def measure_inference_latency(
    model: torch.nn.Module,
    input_length: int,
    device: str = "cpu",
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> dict[str, float]:
    """测量推理延迟。

    Args:
        model: 模型
        input_length: 输入波形长度
        device: 测量设备（cpu/cuda）
        num_runs: 测量次数
        warmup_runs: 预热次数

    Returns:
        包含 mean_ms、std_ms、min_ms、max_ms 的字典
    """
    import time

    model = model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        device_obj = torch.device("cuda")
    else:
        model = model.cpu()
        device_obj = torch.device("cpu")

    example = torch.randn(1, input_length, device=device_obj)

    # 预热
    with torch.no_grad():
        for _ in range(warmup_runs):
            model(example)

    # 同步 GPU
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    # 测量
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(example)
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # 转换为 ms

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "device": device,
        "num_runs": num_runs,
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
    parser.add_argument(
        "--measure-latency",
        action="store_true",
        help="测量推理延迟",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="推理延迟测量设备",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="推理延迟测量次数",
    )
    args = parser.parse_args()

    model, _ = build_model(args.config)
    stats = summarize_efficiency(model, args.input_length)
    for key, value in stats.items():
        print(f"{key}: {value}")

    if args.measure_latency:
        print(f"\n测量推理延迟 (device={args.device}, runs={args.num_runs})...")
        latency = measure_inference_latency(
            model,
            args.input_length,
            device=args.device,
            num_runs=args.num_runs,
        )
        print(f"mean_latency: {latency['mean_ms']:.2f} ms")
        print(f"std_latency: {latency['std_ms']:.2f} ms")
        print(f"min_latency: {latency['min_ms']:.2f} ms")
        print(f"max_latency: {latency['max_ms']:.2f} ms")


if __name__ == "__main__":
    main()
