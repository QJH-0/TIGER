from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_experiments


def test_stage2_mapping_exposes_d0_baseline_slot():
    assert "D0-Baseline" in analyze_experiments.EXPERIMENTS["stage2"]


def test_pick_b_star_stage1_key_uses_lowest_val_loss():
    results = {
        "B1-BN-Mask": 1.2,
        "B2-BN-Mask-DW": 1.1,
        "B3-BN-Mask-DW-PW": 1.05,
        "B4-Full": 1.3,
    }
    assert analyze_experiments.pick_b_star_stage1_key(results) == "B3-BN-Mask-DW-PW"

    results["B2-BN-Mask-DW"] = 0.98
    assert analyze_experiments.pick_b_star_stage1_key(results) == "B2-BN-Mask-DW"


def test_pick_b_star_stage1_key_ignores_missing_values():
    results = {
        "B1-BN-Mask": None,
        "B2-BN-Mask-DW": None,
        "B3-BN-Mask-DW-PW": 1.05,
    }
    assert analyze_experiments.pick_b_star_stage1_key(results) == "B3-BN-Mask-DW-PW"
