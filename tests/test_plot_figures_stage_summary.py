from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plot_figures


def test_stage_summary_function_registry_contains_all_stages():
    registry = plot_figures.get_stage_summary_plot_functions()
    assert set(registry.keys()) == {"stage0", "stage1", "stage2", "stage3"}


def test_stage_summary_output_names_are_stable():
    assert plot_figures.stage_summary_output_name("stage0") == "stage0_summary.png"
    assert plot_figures.stage_summary_output_name("stage1") == "stage1_summary.png"
    assert plot_figures.stage_summary_output_name("stage2") == "stage2_summary.png"
    assert plot_figures.stage_summary_output_name("stage3") == "stage3_summary.png"


def test_pick_available_chinese_font_prefers_first_installed_candidate():
    chosen = plot_figures.pick_available_chinese_font(
        ["Missing Font", "SimHei", "Microsoft YaHei"],
        {"SimHei", "Microsoft YaHei"},
    )
    assert chosen == "SimHei"


def test_pick_available_chinese_font_returns_none_when_no_candidate_exists():
    chosen = plot_figures.pick_available_chinese_font(
        ["Missing Font A", "Missing Font B"],
        {"Microsoft YaHei", "SimHei"},
    )
    assert chosen is None
