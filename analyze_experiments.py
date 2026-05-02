"""实验结果分析工具。

读取各阶段实验的 history.csv / best_model.pth，自动对比 val_loss，
生成汇总报告。

用法：
    python analyze_experiments.py                          # 分析所有实验
    python analyze_experiments.py --phase stage0           # 只分析阶段零
    python analyze_experiments.py --phase stage1           # 只分析阶段一
    python analyze_experiments.py --baseline Experiments/TIGER-Small-MiniLibriMix-Kaggle-T4x2

各阶段目录名与 configs 中 exp.exp_name 一致；Kaggle 与本地同名实验会优先使用已存在的目录。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def _exp_dir_candidates(*paths: str) -> list[str]:
    """返回候选目录列表（Kaggle 与本地 exp_name 可能不同）。"""
    return list(paths)


def _resolve_exp_dir(paths: str | list[str]) -> str:
    """在多个候选路径中选取第一个已存在的目录；均不存在时返回首选路径。"""
    if isinstance(paths, str):
        candidates = [paths]
    else:
        candidates = paths
    if not candidates:
        return ""
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


# 与《综合实验方案》叙事对齐：B* 默认指阶段一中的「激进组合」B3（BN+mask+DW+PW），作蒸馏与消融的主线二值基线。
# B0 表示「全量二值」单独配置；若以 B0 为蒸馏入口请在论文/表格中显式写清，避免与 B* 混用。
B_STAR_FALLBACK_STAGE1_KEY = "B3-BN-Mask-DW-PW"

# 实验目录映射（值为单个路径或候选路径列表，与 configs 中 exp_name 一致）
EXPERIMENTS = {
    # 前置：全精度
    "fp32": _exp_dir_candidates(
        "Experiments/TIGER-Small-MiniLibriMix-Kaggle-T4x2",
        "Experiments/TIGER-MiniLibriMix-Local",
    ),
    # 阶段零：敏感度验证
    "stage0": {
        "S0-BN": _exp_dir_candidates(
            "Experiments/TIGER-Small-Binary-S0-BN-Kaggle-T4x2",
            "Experiments/TIGER-Small-Binary-S0-BN",
        ),
        "S1-Mask": _exp_dir_candidates(
            "Experiments/TIGER-Small-Binary-S1-Mask-Kaggle-T4x2",
            "Experiments/TIGER-Small-Binary-S1-Mask",
        ),
        "S2-DW": _exp_dir_candidates(
            "Experiments/TIGER-Small-Binary-S2-DW-Kaggle-T4x2",
            "Experiments/TIGER-Small-Binary-S2-DW",
        ),
        "S3-PW": _exp_dir_candidates(
            "Experiments/TIGER-Small-Binary-S3-PW-Kaggle-T4x2",
            "Experiments/TIGER-Small-Binary-S3-PW",
        ),
        "S4-F3A": _exp_dir_candidates(
            "Experiments/TIGER-Small-Binary-S4-F3A-Kaggle-T4x2",
            "Experiments/TIGER-Small-Binary-S4-F3A",
        ),
    },
    # 阶段一：组合二值化
    "stage1": {
        "B0-Full": _exp_dir_candidates(
            "Experiments/TIGER-Small-Binary-B0-Kaggle-T4x2",
        ),
        "B1-BN-Mask": _exp_dir_candidates(
            "Experiments/TIGER-Small-Binary-B1-BN-Mask-Kaggle-T4x2",
            "Experiments/TIGER-Small-Binary-B1-BN-Mask",
        ),
        "B2-BN-Mask-DW": _exp_dir_candidates(
            "Experiments/TIGER-Small-Binary-B2-BN-Mask-DW-Kaggle-T4x2",
            "Experiments/TIGER-Small-Binary-B2-BN-Mask-DW",
        ),
        "B3-BN-Mask-DW-PW": _exp_dir_candidates(
            "Experiments/TIGER-Small-Binary-B3-BN-Mask-DW-PW-Kaggle-T4x2",
            "Experiments/TIGER-Small-Binary-B3-BN-Mask-DW-PW",
        ),
        "B4-Full": _exp_dir_candidates(
            "Experiments/TIGER-Small-Binary-B4-Full-Kaggle-T4x2",
            "Experiments/TIGER-Small-Binary-B4-Full",
        ),
    },
    # 阶段二：蒸馏补偿
    "stage2": {
        "D0-Baseline": [],
        "D1-Output": _exp_dir_candidates(
            "Experiments/TIGER-Small-Binary-Distill-D1-Kaggle-T4x2",
            "Experiments/TIGER-Small-Binary-Distill-D1-Local",
        ),
        "D2-Subband": _exp_dir_candidates(
            "Experiments/TIGER-Small-Binary-Distill-D2-Kaggle-T4x2",
            "Experiments/TIGER-Small-Binary-Distill-D2-Local",
        ),
        "D3-Combined": _exp_dir_candidates(
            "Experiments/TIGER-Small-Binary-Distill-Kaggle-T4x2",
            "Experiments/TIGER-Small-Binary-Distill-D3-Local",
        ),
    },
    # 阶段三：消融实验
    "stage3": {
        "A1-NoRSign": _exp_dir_candidates(
            "Experiments/TIGER-Small-Binary-A1-NoRSign-Kaggle-T4x2",
            "Experiments/TIGER-Small-Binary-A1-NoRSign",
        ),
        "A2-NoWarmup": _exp_dir_candidates(
            "Experiments/TIGER-Small-Binary-A2-NoWarmup-Kaggle-T4x2",
            "Experiments/TIGER-Small-Binary-A2-NoWarmup",
        ),
        "A3-PReLU": _exp_dir_candidates(
            "Experiments/TIGER-Small-Binary-A3-PReLU-Kaggle-T4x2",
            "Experiments/TIGER-Small-Binary-A3-PReLU",
        ),
        "A4-NoSubbandWeight": _exp_dir_candidates(
            "Experiments/TIGER-Small-Binary-A4-NoSubbandWeight-Kaggle-T4x2",
            "Experiments/TIGER-Small-Binary-A4-NoSubbandWeight",
        ),
    },
}


def collect_stage1_best_losses() -> dict[str, float | None]:
    losses: dict[str, float | None] = {}
    for key, exp_paths in EXPERIMENTS["stage1"].items():
        exp_dir = _resolve_exp_dir(exp_paths)
        losses[key] = read_best_val_loss(exp_dir) if os.path.exists(exp_dir) else None
    return losses


def pick_b_star_stage1_key(stage1_losses: dict[str, float | None]) -> str:
    available = [(key, loss) for key, loss in stage1_losses.items() if loss is not None]
    if not available:
        return B_STAR_FALLBACK_STAGE1_KEY
    return min(available, key=lambda item: item[1])[0]


def resolve_d0_candidates() -> list[str]:
    stage1_losses = collect_stage1_best_losses()
    b_star_key = pick_b_star_stage1_key(stage1_losses)
    return list(EXPERIMENTS["stage1"].get(b_star_key, []))


def read_history_csv(exp_dir: str | list[str]) -> list[dict]:
    """读取 history.csv 文件。"""
    exp_dir = _resolve_exp_dir(exp_dir)
    if not exp_dir:
        return []
    csv_path = os.path.join(exp_dir, "history.csv")
    if not os.path.exists(csv_path):
        return []
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def read_best_val_loss(exp_dir: str) -> float | None:
    """从 history.csv 中读取最佳 val_loss。"""
    rows = read_history_csv(exp_dir)
    if not rows:
        return None
    val_losses = []
    for row in rows:
        val_loss = row.get("val_loss", "")
        if val_loss:
            try:
                val_losses.append(float(val_loss))
            except ValueError:
                pass
    return min(val_losses) if val_losses else None


def read_final_val_loss(exp_dir: str) -> float | None:
    """从 history.csv 中读取最终 val_loss。"""
    rows = read_history_csv(exp_dir)
    if not rows:
        return None
    for row in reversed(rows):
        val_loss = row.get("val_loss", "")
        if val_loss:
            try:
                return float(val_loss)
            except ValueError:
                pass
    return None


def compute_increase_pct(val_loss: float, baseline: float) -> float:
    """计算 val_loss 相对基线的上升百分比。"""
    if baseline == 0:
        return 0.0
    return (val_loss - baseline) / abs(baseline) * 100.0


def analyze_phase(phase_name: str, experiments: dict, baseline_val_loss: float | None) -> None:
    """分析单个阶段的实验结果。"""
    print(f"\n{'='*60}")
    print(f"  {phase_name}")
    if "组合二值化" in phase_name:
        print(f"  （B* 缺省回退：{B_STAR_FALLBACK_STAGE1_KEY}；若存在真实结果则自动取阶段一最优）")
    print(f"{'='*60}")

    results = []
    for name, exp_paths in experiments.items():
        if name == "D0-Baseline":
            exp_paths = resolve_d0_candidates()
            if not exp_paths:
                results.append((name, "NOT FOUND", None, None))
                continue
        exp_dir = _resolve_exp_dir(exp_paths)
        if not os.path.exists(exp_dir):
            results.append((name, "NOT FOUND", None, None))
            continue

        best_loss = read_best_val_loss(exp_dir)
        final_loss = read_final_val_loss(exp_dir)
        history = read_history_csv(exp_dir)
        epochs = len(history)

        if best_loss is None:
            results.append((name, "NO DATA", None, None))
            continue

        increase = None
        if baseline_val_loss is not None and baseline_val_loss > 0:
            increase = compute_increase_pct(best_loss, baseline_val_loss)

        results.append((name, "OK", best_loss, increase, final_loss, epochs))

    # 打印表格
    print(f"\n{'实验':<25} {'状态':<10} {'best_val_loss':<15} {'vs基线':<10} {'final_val_loss':<15} {'epochs':<8}")
    print("-" * 85)

    for item in results:
        name = item[0]
        status = item[1]
        if status != "OK":
            print(f"{name:<25} {status:<10}")
        else:
            best = item[2]
            increase = item[3]
            final = item[4]
            epochs = item[5]

            inc_str = f"{increase:+.1f}%" if increase is not None else "N/A"
            final_str = f"{final:.4f}" if final is not None else "N/A"

            print(f"{name:<25} {'OK':<10} {best:<15.4f} {inc_str:<10} {final_str:<15} {epochs:<8}")

    # 返回阶段零的最佳结果（用于阶段一的基线）
    if phase_name.startswith("阶段零"):
        best_stage0 = min(
            [item[2] for item in results if item[1] == "OK" and item[2] is not None],
            default=None,
        )
        return best_stage0
    return None


def read_final_metrics(exp_dir: str) -> dict | None:
    """读取 final_metrics.json 文件。"""
    metrics_path = os.path.join(exp_dir, "final_metrics.json")
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _metric_value(metrics: dict, *keys: str):
    for key in keys:
        if key in metrics:
            return metrics[key]
    return "—"


def generate_main_table(baseline_val_loss: float | None = None) -> str:
    """生成主结果表格（Markdown 格式）。

    汇总所有实验的 SI-SNRi、SDR、PESQ、STOI、模型大小、BOPs。
    """
    lines = []
    lines.append("| 模型 | SI-SNRi | SDR | PESQ | STOI | 模型大小 | BOPs | 相对 FP32 |")
    lines.append("|------|---------|-----|------|------|---------|------|----------|")

    # FP32 基线
    fp32_dir = _resolve_exp_dir(EXPERIMENTS["fp32"])
    fp32_metrics = read_final_metrics(fp32_dir)
    fp32_sisnri = None
    if fp32_metrics and "metrics" in fp32_metrics:
        m = fp32_metrics["metrics"]
        fp32_sisnri = _metric_value(m, "si_snri", "si_snr_i")
        lines.append(
            f"| **FP32** | **{_metric_value(m, 'si_snri', 'si_snr_i')}** | **{_metric_value(m, 'sdr')}** | "
            f"**{m.get('pesq', '—')}** | **{m.get('stoi', '—')}** | "
            f"**{fp32_metrics.get('efficiency', {}).get('fp32_estimated_size_mb', '—')}MB** | "
            f"**{fp32_metrics.get('efficiency', {}).get('bops', fp32_metrics.get('efficiency', {}).get('binary_bops', '—'))}G** | **—** |"
        )
    else:
        lines.append("| **FP32** | — | — | — | — | — | — | — |")

    # 阶段一：组合二值化
    stage1 = EXPERIMENTS["stage1"]
    stage1_order = ["B1-BN-Mask", "B2-BN-Mask-DW", "B3-BN-Mask-DW-PW"]
    b_star_key = pick_b_star_stage1_key(collect_stage1_best_losses())
    for key in stage1_order:
        if key not in stage1:
            continue
        exp_dir = _resolve_exp_dir(stage1[key])
        metrics = read_final_metrics(exp_dir)
        best_loss = read_best_val_loss(exp_dir)

        if metrics and "metrics" in metrics:
            m = metrics["metrics"]
            sisnri = _metric_value(m, "si_snri", "si_snr_i")
            sdr = _metric_value(m, "sdr")
            pesq = m.get("pesq", "—")
            stoi = m.get("stoi", "—")
            size = metrics.get("efficiency", {}).get("fp32_estimated_size_mb", "—")
            bops = metrics.get("efficiency", {}).get("bops", metrics.get("efficiency", {}).get("binary_bops", "—"))

            # 计算相对 FP32 的变化
            rel_fp32 = "—"
            if fp32_sisnri is not None and isinstance(sisnri, (int, float)):
                rel_fp32 = f"{sisnri - fp32_sisnri:+.1f} dB"

            # 标记 B*
            name = key.split("-")[0]
            if key == b_star_key:
                name = f"B* (={name})"

            lines.append(
                f"| {name} | {sisnri} | {sdr} | {pesq} | {stoi} | "
                f"{size}MB | {bops}G | {rel_fp32} |"
            )
        else:
            name = key.split("-")[0]
            if key == b_star_key:
                name = f"B* (={name})"
            lines.append(f"| {name} | — | — | — | — | — | — | — |")

    # 阶段二：蒸馏补偿
    stage2 = EXPERIMENTS["stage2"]
    stage2_order = ["D0-Baseline", "D1-Output", "D2-Subband", "D3-Combined"]
    for key in stage2_order:
        exp_paths = resolve_d0_candidates() if key == "D0-Baseline" else stage2.get(key)
        if not exp_paths:
            continue
        exp_dir = _resolve_exp_dir(exp_paths)
        metrics = read_final_metrics(exp_dir)

        if metrics and "metrics" in metrics:
            m = metrics["metrics"]
            sisnri = _metric_value(m, "si_snri", "si_snr_i")
            sdr = _metric_value(m, "sdr")
            pesq = m.get("pesq", "—")
            stoi = m.get("stoi", "—")
            size = metrics.get("efficiency", {}).get("fp32_estimated_size_mb", "—")
            bops = metrics.get("efficiency", {}).get("bops", metrics.get("efficiency", {}).get("binary_bops", "—"))

            # 计算相对 FP32 的变化
            rel_fp32 = "—"
            if fp32_sisnri is not None and isinstance(sisnri, (int, float)):
                rel_fp32 = f"{sisnri - fp32_sisnri:+.1f} dB"

            name = "D0" if key == "D0-Baseline" else key.split("-")[0]
            if key == "D3-Combined":
                name = "B* + D3"

            lines.append(
                f"| **{name}** | **{sisnri}** | **{sdr}** | **{pesq}** | **{stoi}** | "
                f"{size}MB | {bops}G | {rel_fp32} |"
            )
        else:
            name = "D0" if key == "D0-Baseline" else key.split("-")[0]
            if key == "D3-Combined":
                name = "B* + D3"
            lines.append(f"| {name} | — | — | — | — | — | — | — |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="TIGER 实验结果分析工具")
    parser.add_argument(
        "--phase",
        choices=["all", "stage0", "stage1", "stage2", "stage3"],
        default="all",
        help="要分析的阶段",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="FP32 基线实验目录（自动读取 best val_loss）",
    )
    parser.add_argument(
        "--baseline-loss",
        type=float,
        default=None,
        help="直接指定 FP32 基线 val_loss",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="生成主结果表格（Markdown 格式）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（与 --table 配合使用）",
    )
    args = parser.parse_args()

    # 确定基线 val_loss
    baseline_val_loss = None
    if args.baseline_loss is not None:
        baseline_val_loss = args.baseline_loss
    elif args.baseline:
        baseline_val_loss = read_best_val_loss(args.baseline)
    else:
        # 尝试默认路径
        baseline_val_loss = read_best_val_loss(EXPERIMENTS["fp32"])

    if baseline_val_loss is not None:
        print(f"FP32 基线 val_loss: {baseline_val_loss:.4f}")
    else:
        print("警告：未找到 FP32 基线 val_loss，将无法计算上升百分比")

    # 分析各阶段
    if args.phase in ("all", "stage0"):
        analyze_phase("阶段零：敏感度验证", EXPERIMENTS["stage0"], baseline_val_loss)

    if args.phase in ("all", "stage1"):
        analyze_phase("阶段一：组合二值化", EXPERIMENTS["stage1"], baseline_val_loss)

    if args.phase in ("all", "stage2"):
        analyze_phase("阶段二：蒸馏补偿", EXPERIMENTS["stage2"], baseline_val_loss)

    if args.phase in ("all", "stage3"):
        analyze_phase("阶段三：消融实验", EXPERIMENTS["stage3"], baseline_val_loss)

    # 生成 JSON 汇总
    if args.phase == "all":
        summary = {
            "baseline_val_loss": baseline_val_loss,
            "experiments": {},
        }
        for phase_name, phase_exps in [
            ("stage0", EXPERIMENTS["stage0"]),
            ("stage1", EXPERIMENTS["stage1"]),
            ("stage2", EXPERIMENTS["stage2"]),
            ("stage3", EXPERIMENTS["stage3"]),
        ]:
            summary["experiments"][phase_name] = {}
            for name, exp_paths in phase_exps.items():
                if name == "D0-Baseline":
                    exp_paths = resolve_d0_candidates()
                exp_dir = _resolve_exp_dir(exp_paths)
                best = read_best_val_loss(exp_dir)
                final = read_final_val_loss(exp_dir)
                increase = None
                if best is not None and baseline_val_loss is not None and baseline_val_loss > 0:
                    increase = compute_increase_pct(best, baseline_val_loss)
                summary["experiments"][phase_name][name] = {
                    "dir": exp_dir,
                    "best_val_loss": best,
                    "final_val_loss": final,
                    "increase_pct": increase,
                }

        output_path = "experiment_summary.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n汇总报告已保存到: {output_path}")

    # 生成主结果表格
    if args.table:
        table = generate_main_table(baseline_val_loss)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write("# TIGER 主实验结果\n\n")
                f.write(table)
                f.write("\n")
            print(f"\n主结果表格已保存到: {args.output}")
        else:
            print("\n# TIGER 主实验结果\n")
            print(table)


if __name__ == "__main__":
    main()
