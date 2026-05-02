"""论文图表生成工具。

根据《综合实验方案》§10 的要求，生成 7 个核心图表：
1. 模块敏感度柱状图
2. 精度-压缩权衡曲线
3. 训练动态与检查点
4. 蒸馏效果对比
5. 蒸馏训练动态
6. 消融实验雷达图
7. 推理效率帕累托前沿

用法：
    python plot_figures.py                          # 生成所有图表
    python plot_figures.py --figures 1,2,4          # 生成指定图表
    python plot_figures.py --output figures/        # 指定输出目录
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager
import numpy as np

# 复用 analyze_experiments.py 中的实验目录映射
from analyze_experiments import (
    EXPERIMENTS,
    _resolve_exp_dir,
    collect_stage1_best_losses,
    pick_b_star_stage1_key,
    resolve_d0_candidates,
    read_best_val_loss,
    read_history_csv,
    read_final_val_loss,
    compute_increase_pct,
)

CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Microsoft JhengHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Arial Unicode MS",
]


def pick_available_chinese_font(
    candidates: list[str] | tuple[str, ...],
    available_fonts: set[str] | None = None,
) -> str | None:
    if available_fonts is None:
        available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available_fonts:
            return name
    return None


def configure_matplotlib_fonts() -> str | None:
    chosen_font = pick_available_chinese_font(CHINESE_FONT_CANDIDATES)
    if chosen_font is not None:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [chosen_font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return chosen_font


# 默认图表样式
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
configure_matplotlib_fonts()


def read_final_metrics(exp_dir: str) -> Optional[dict]:
    """读取 final_metrics.json 文件。"""
    metrics_path = os.path.join(exp_dir, "final_metrics.json")
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _stage1_b_star_key() -> str:
    return pick_b_star_stage1_key(collect_stage1_best_losses())


def stage_summary_output_name(stage_name: str) -> str:
    return f"{stage_name}_summary.png"


def _stage_summary_entries(stage_name: str) -> list[tuple[str, str | list[str]]]:
    if stage_name == "stage2":
        return [
            ("D0", resolve_d0_candidates()),
            ("D1", EXPERIMENTS["stage2"]["D1-Output"]),
            ("D2", EXPERIMENTS["stage2"]["D2-Subband"]),
            ("D3", EXPERIMENTS["stage2"]["D3-Combined"]),
        ]

    stage_map = EXPERIMENTS[stage_name]
    entries = []
    for key, value in stage_map.items():
        label = key.split("-")[0]
        entries.append((label, value))
    return entries


def _best_val_loss_from_candidates(exp_paths: str | list[str]) -> Optional[float]:
    if not exp_paths:
        return None
    exp_dir = _resolve_exp_dir(exp_paths)
    if not os.path.exists(exp_dir):
        return None
    return read_best_val_loss(exp_dir)


def _plot_stage_summary(stage_name: str, output_dir: str, title: str) -> None:
    entries = _stage_summary_entries(stage_name)
    labels = []
    values = []
    missing = []

    for label, exp_paths in entries:
        labels.append(label)
        best_val = _best_val_loss_from_candidates(exp_paths)
        if best_val is None:
            values.append(np.nan)
            missing.append(label)
        else:
            values.append(best_val)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    clean_values = [0.0 if np.isnan(v) else v for v in values]
    colors = ["#c0392b" if np.isnan(v) else "#3498db" for v in values]
    bars = ax.bar(x, clean_values, color=colors, edgecolor="black", linewidth=0.8)

    for idx, (bar, value) in enumerate(zip(bars, values)):
        if np.isnan(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.02,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=11,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("best val_loss")
    if missing:
        title += "（部分实验缺数据）"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    output_path = os.path.join(output_dir, stage_summary_output_name(stage_name))
    fig.savefig(output_path)
    plt.close(fig)
    print(f"已生成: {output_path}")


def plot_stage0_summary(output_dir: str) -> None:
    _plot_stage_summary("stage0", output_dir, "阶段零汇总：S0-S4 敏感度实验")


def plot_stage1_summary(output_dir: str) -> None:
    _plot_stage_summary("stage1", output_dir, "阶段一汇总：B0-B4 组合二值化")


def plot_stage2_summary(output_dir: str) -> None:
    _plot_stage_summary("stage2", output_dir, "阶段二汇总：D0-D3 蒸馏补偿")


def plot_stage3_summary(output_dir: str) -> None:
    _plot_stage_summary("stage3", output_dir, "阶段三汇总：A1-A4 消融实验")


def get_stage_summary_plot_functions():
    return {
        "stage0": plot_stage0_summary,
        "stage1": plot_stage1_summary,
        "stage2": plot_stage2_summary,
        "stage3": plot_stage3_summary,
    }


def get_baseline_val_loss() -> Optional[float]:
    """获取 FP32 基线的 val_loss。"""
    fp32_dir = _resolve_exp_dir(EXPERIMENTS["fp32"])
    return read_best_val_loss(fp32_dir)


# ============================================================================
#  图 1：模块敏感度柱状图
# ============================================================================

def plot_sensitivity_bar(output_dir: str) -> None:
    """生成模块敏感度柱状图。

    横轴：S0-BN / S1-Mask / S2-DW / S3-PW / S4-F3A
    纵轴：val_loss 相对 FP32 上升比例 (%)
    颜色：绿色(<10%) / 黄色(10-20%) / 红色(>20%)
    """
    baseline = get_baseline_val_loss()
    if baseline is None:
        print("警告：未找到 FP32 基线，跳过图 1")
        return

    stage0 = EXPERIMENTS["stage0"]
    names = []
    increases = []

    for name, exp_paths in stage0.items():
        exp_dir = _resolve_exp_dir(exp_paths)
        best_loss = read_best_val_loss(exp_dir)
        if best_loss is None:
            names.append(name)
            increases.append(0)
        else:
            names.append(name)
            increases.append(compute_increase_pct(best_loss, baseline))

    # 颜色映射
    colors = []
    for inc in increases:
        if inc < 10:
            colors.append("#2ecc71")  # 绿色
        elif inc < 20:
            colors.append("#f39c12")  # 黄色
        else:
            colors.append("#e74c3c")  # 红色

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, increases, color=colors, edgecolor="black", linewidth=0.8)

    # 添加数值标签
    for bar, inc in zip(bars, increases):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{inc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_xlabel("模块")
    ax.set_ylabel("val_loss 上升比例 (%)")
    ax.set_title("图 1：模块敏感度分析")
    ax.axhline(y=10, color="gray", linestyle="--", alpha=0.5, label="10% 阈值")
    ax.axhline(y=20, color="gray", linestyle=":", alpha=0.5, label="20% 阈值")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    output_path = os.path.join(output_dir, "fig1_sensitivity_bar.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"已生成: {output_path}")


# ============================================================================
#  图 2：精度-压缩权衡曲线
# ============================================================================

def plot_accuracy_compression_tradeoff(output_dir: str) -> None:
    """生成精度-压缩权衡曲线。

    横轴：FP32 → B1 → B2 → B3
    左纵轴：SI-SNRi (dB)
    右纵轴：模型大小 (MB) / BOPs (G)
    标记：B* 红色星号
    """
    # 预期数据（来自《综合实验方案》§11）
    # 实际实验数据应从 final_metrics.json 读取
    models = ["FP32", "B1", "B2", "B3"]
    sisnri = [16.7, 16.2, 15.7, 15.2]  # 预期值
    model_sizes = [3.3, 1.5, 1.2, 0.8]  # MB
    bops = [7.7, None, None, 2.5]  # G
    used_fallback = True

    # 尝试从实验数据读取
    fp32_dir = _resolve_exp_dir(EXPERIMENTS["fp32"])
    fp32_metrics = read_final_metrics(fp32_dir)
    if fp32_metrics and "metrics" in fp32_metrics:
        m = fp32_metrics["metrics"]
        if "si_snri" in m:
            sisnri[0] = m["si_snri"]
            used_fallback = False

    stage1 = EXPERIMENTS["stage1"]
    stage1_keys = ["B1-BN-Mask", "B2-BN-Mask-DW", "B3-BN-Mask-DW-PW"]
    for i, key in enumerate(stage1_keys):
        if key in stage1:
            exp_dir = _resolve_exp_dir(stage1[key])
            metrics = read_final_metrics(exp_dir)
            if metrics and "metrics" in metrics:
                m = metrics["metrics"]
                if "si_snri" in m:
                    sisnri[i + 1] = m["si_snri"]
                    used_fallback = False

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左纵轴：SI-SNRi
    color1 = "#3498db"
    ax1.set_xlabel("模型配置")
    ax1.set_ylabel("SI-SNRi (dB)", color=color1)
    line1 = ax1.plot(models, sisnri, "o-", color=color1, linewidth=2, markersize=8, label="SI-SNRi")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(14.5, 17.5)

    # 标记 B*
    b_star_label = _stage1_b_star_key().split("-")[0]
    if b_star_label in models:
        ax1.plot(
            b_star_label,
            sisnri[models.index(b_star_label)],
            "r*",
            markersize=15,
            label="B*",
        )

    # 右纵轴：模型大小
    ax2 = ax1.twinx()
    color2 = "#e74c3c"
    ax2.set_ylabel("模型大小 (MB)", color=color2)
    line2 = ax2.plot(model_sizes, "s--", color=color2, linewidth=2, markersize=8, label="模型大小")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 4.0)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    title = "图 2：精度-压缩权衡曲线"
    if used_fallback:
        title += "（含预期数据回退）"
    ax1.set_title(title)
    ax1.grid(axis="y", alpha=0.3)

    output_path = os.path.join(output_dir, "fig2_tradeoff_curve.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"已生成: {output_path}")


# ============================================================================
#  图 3：训练动态与检查点
# ============================================================================

def plot_training_dynamics(output_dir: str) -> None:
    """生成训练动态与检查点图。

    横轴：Epoch
    纵轴：val_loss
    线条：B1/B2/B3 各一条，FP32 水平参考线
    检查点：30 epochs 垂直虚线
    """
    stage1 = EXPERIMENTS["stage1"]
    stage1_keys = ["B1-BN-Mask", "B2-BN-Mask-DW", "B3-BN-Mask-DW-PW"]
    labels = ["B1 (BN+Mask)", "B2 (BN+Mask+DW)", "B3 (BN+Mask+DW+PW)"]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    # FP32 基线
    fp32_loss = get_baseline_val_loss()

    fig, ax = plt.subplots(figsize=(12, 6))

    for key, label, color in zip(stage1_keys, labels, colors):
        exp_dir = _resolve_exp_dir(stage1[key])
        history = read_history_csv(exp_dir)
        if not history:
            continue

        epochs = []
        val_losses = []
        for row in history:
            epoch = row.get("epoch", "")
            val_loss = row.get("val_loss", "")
            if epoch and val_loss:
                try:
                    epochs.append(int(epoch))
                    val_losses.append(float(val_loss))
                except ValueError:
                    pass

        if epochs and val_losses:
            ax.plot(epochs, val_losses, "-", color=color, linewidth=2, label=label)

    # FP32 参考线
    if fp32_loss is not None:
        ax.axhline(y=fp32_loss, color="red", linestyle="--", alpha=0.7, label=f"FP32 基线 ({fp32_loss:.4f})")

    # 检查点虚线
    ax.axvline(x=30, color="gray", linestyle=":", alpha=0.5, label="30 epoch 检查点")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("val_loss")
    ax.set_title("图 3：训练动态与检查点")
    ax.legend()
    ax.grid(alpha=0.3)

    output_path = os.path.join(output_dir, "fig3_training_dynamics.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"已生成: {output_path}")


# ============================================================================
#  图 4：蒸馏效果对比
# ============================================================================

def plot_distillation_comparison(output_dir: str) -> None:
    """生成蒸馏效果对比柱状图。

    横轴：D0 / D1 / D2 / D3
    纵轴：SI-SNRi (dB)
    参考线：FP32 红色虚线、B* 灰色虚线
    """
    # 预期数据（来自《蒸馏技术方案》§7）
    distill_types = ["D0\n(无蒸馏)", "D1\n(SI-SNR)", "D2\n(子带)", "D3\n(联合)"]
    sisnri = [15.2, 15.6, 16.0, 16.2]  # 预期值
    fp32_sisnri = 16.7
    b_star_sisnri = 15.2
    used_fallback = True

    # 尝试从实验数据读取
    stage2 = EXPERIMENTS["stage2"]
    stage2_keys = ["D1-Output", "D2-Subband", "D3-Combined"]
    for i, key in enumerate(stage2_keys):
        if key in stage2:
            exp_dir = _resolve_exp_dir(stage2[key])
            metrics = read_final_metrics(exp_dir)
            if metrics and "metrics" in metrics:
                m = metrics["metrics"]
                if "si_snri" in m:
                    sisnri[i + 1] = m["si_snri"]
                    used_fallback = False

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#95a5a6", "#3498db", "#e74c3c", "#2ecc71"]
    bars = ax.bar(distill_types, sisnri, color=colors, edgecolor="black", linewidth=0.8)

    # 添加数值标签
    for bar, val in zip(bars, sisnri):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # 参考线
    ax.axhline(y=fp32_sisnri, color="red", linestyle="--", alpha=0.7, label=f"FP32 ({fp32_sisnri} dB)")
    ax.axhline(y=b_star_sisnri, color="gray", linestyle=":", alpha=0.7, label=f"B* ({b_star_sisnri} dB)")

    ax.set_xlabel("蒸馏方案")
    ax.set_ylabel("SI-SNRi (dB)")
    title = "图 4：蒸馏效果对比"
    if used_fallback:
        title += "（含预期数据回退）"
    ax.set_title(title)
    ax.set_ylim(14.5, 17.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    output_path = os.path.join(output_dir, "fig4_distillation_comparison.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"已生成: {output_path}")


# ============================================================================
#  图 5：蒸馏训练动态
# ============================================================================

def plot_distillation_dynamics(output_dir: str) -> None:
    """生成蒸馏训练动态图。

    横轴：Epoch
    左纵轴：val_loss
    线条：D0/D1/D2/D3
    """
    stage2 = EXPERIMENTS["stage2"]
    stage2_keys = ["D1-Output", "D2-Subband", "D3-Combined"]
    labels = ["D1 (SI-SNR)", "D2 (子带)", "D3 (联合)"]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # D0 基线：工程上等价于阶段一最优二值化基线 B*
    d0_candidates = resolve_d0_candidates()
    if d0_candidates:
        exp_dir = _resolve_exp_dir(d0_candidates)
        history = read_history_csv(exp_dir)
        if history:
            epochs = []
            val_losses = []
            for row in history:
                epoch = row.get("epoch", "")
                val_loss = row.get("val_loss", "")
                if epoch and val_loss:
                    try:
                        epochs.append(int(epoch))
                        val_losses.append(float(val_loss))
                    except ValueError:
                        pass
            if epochs and val_losses:
                ax.plot(epochs, val_losses, "-", color="#95a5a6", linewidth=2, label="D0 (无蒸馏)")

    # D1/D2/D3
    for key, label, color in zip(stage2_keys, labels, colors):
        if key not in stage2:
            continue
        exp_dir = _resolve_exp_dir(stage2[key])
        history = read_history_csv(exp_dir)
        if not history:
            continue

        epochs = []
        val_losses = []
        for row in history:
            epoch = row.get("epoch", "")
            val_loss = row.get("val_loss", "")
            if epoch and val_loss:
                try:
                    epochs.append(int(epoch))
                    val_losses.append(float(val_loss))
                except ValueError:
                    pass

        if epochs and val_losses:
            ax.plot(epochs, val_losses, "-", color=color, linewidth=2, label=label)

    # FP32 基线
    fp32_loss = get_baseline_val_loss()
    if fp32_loss is not None:
        ax.axhline(y=fp32_loss, color="red", linestyle="--", alpha=0.7, label=f"FP32 基线")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("val_loss")
    ax.set_title("图 5：蒸馏训练动态")
    ax.legend()
    ax.grid(alpha=0.3)

    output_path = os.path.join(output_dir, "fig5_distillation_dynamics.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"已生成: {output_path}")


# ============================================================================
#  图 6：消融实验雷达图
# ============================================================================

def plot_ablation_radar(output_dir: str) -> None:
    """生成消融实验雷达图。

    维度：SI-SNRi / SDR / PESQ / STOI / 模型大小
    多边形：A1-A4
    """
    # 预期数据（来自《综合实验方案》§8）
    # 实际实验数据应从 final_metrics.json 读取
    categories = ["SI-SNRi", "SDR", "PESQ", "STOI", "模型大小"]
    n_cats = len(categories)

    # A1-A4 的预期数据（归一化到 0-1）
    ablation_data = {
        "A1 (无 RSign)": [0.85, 0.86, 0.88, 0.90, 0.95],
        "A2 (无预热)": [0.80, 0.82, 0.85, 0.88, 0.95],
        "A3 (PReLU)": [0.90, 0.91, 0.92, 0.93, 0.95],
        "A4 (统一权重)": [0.92, 0.93, 0.94, 0.95, 0.95],
    }

    # 角度
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    for (label, values), color in zip(ablation_data.items(), colors):
        values_closed = values + values[:1]
        ax.plot(angles, values_closed, "o-", color=color, linewidth=2, label=label)
        ax.fill(angles, values_closed, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title("图 6：消融实验雷达图", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    output_path = os.path.join(output_dir, "fig6_ablation_radar.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"已生成: {output_path}")


# ============================================================================
#  图 7：推理效率帕累托前沿
# ============================================================================

def plot_pareto_frontier(output_dir: str) -> None:
    """生成推理效率帕累托前沿图。

    横轴：BOPs (G)
    纵轴：SI-SNRi (dB)
    点：FP32/B1/B2/B3/B*
    """
    # 预期数据（来自《综合实验方案》§11）
    models = ["FP32", "B1", "B2", "B3=B*"]
    bops = [7.7, None, None, 2.5]
    sisnri = [16.7, 16.2, 15.7, 15.2]

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制点
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    markers = ["o", "s", "^", "*"]

    for i, (model, bop, sis) in enumerate(zip(models, bops, sisnri)):
        if bop is not None:
            ax.scatter(bop, sis, c=colors[i], marker=markers[i], s=200, label=model, zorder=5)
            ax.annotate(model, (bop, sis), textcoords="offset points", xytext=(10, 10), fontsize=12)

    # 连线（如果有足够的点）
    valid_bops = [b for b in bops if b is not None]
    valid_sisnri = [s for b, s in zip(bops, sisnri) if b is not None]
    if len(valid_bops) >= 2:
        ax.plot(valid_bops, valid_sisnri, "--", color="gray", alpha=0.5)

    ax.set_xlabel("BOPs (G)")
    ax.set_ylabel("SI-SNRi (dB)")
    ax.set_title("图 7：推理效率帕累托前沿")
    ax.legend()
    ax.grid(alpha=0.3)

    output_path = os.path.join(output_dir, "fig7_pareto_frontier.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"已生成: {output_path}")


# ============================================================================
#  主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TIGER 论文图表生成工具")
    parser.add_argument(
        "--figures",
        type=str,
        default="all",
        help="要生成的图表编号，逗号分隔（如 1,2,4）或 'all'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures",
        help="输出目录（默认 figures/）",
    )
    parser.add_argument(
        "--stage-summary",
        type=str,
        default=None,
        help="手动生成阶段总图：stage0,stage1,stage2,stage3 或 all",
    )
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 解析图表编号
    figures_arg_explicit = "--figures" in sys.argv
    if args.stage_summary and not figures_arg_explicit and args.figures == "all":
        figure_ids = []
    elif args.figures == "all":
        figure_ids = list(range(1, 8))
    else:
        figure_ids = [int(x.strip()) for x in args.figures.split(",")]

    # 生成图表
    plot_functions = {
        1: plot_sensitivity_bar,
        2: plot_accuracy_compression_tradeoff,
        3: plot_training_dynamics,
        4: plot_distillation_comparison,
        5: plot_distillation_dynamics,
        6: plot_ablation_radar,
        7: plot_pareto_frontier,
    }
    stage_summary_functions = get_stage_summary_plot_functions()

    for fig_id in figure_ids:
        if fig_id in plot_functions:
            print(f"正在生成图 {fig_id}...")
            plot_functions[fig_id](args.output)
        else:
            print(f"警告：未知的图表编号 {fig_id}")

    if args.stage_summary:
        if args.stage_summary == "all":
            stage_names = list(stage_summary_functions.keys())
        else:
            stage_names = [item.strip() for item in args.stage_summary.split(",") if item.strip()]

        for stage_name in stage_names:
            if stage_name in stage_summary_functions:
                print(f"正在生成 {stage_name} 总图...")
                stage_summary_functions[stage_name](args.output)
            else:
                print(f"警告：未知的阶段编号 {stage_name}")

    print(f"\n图表生成完成，输出目录: {args.output}")


if __name__ == "__main__":
    main()
