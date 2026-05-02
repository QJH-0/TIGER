import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_test import update_final_metrics_with_test_summary


def test_update_final_metrics_with_test_summary_merges_metric_aliases(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    metrics_path = exp_dir / "final_metrics.json"
    metrics_path.write_text(
        json.dumps({"metrics": {"val_loss": 0.5}, "efficiency": {"fp32_estimated_size_mb": 3.2}}),
        encoding="utf-8",
    )

    summary = {
        "test/sdr": 12.3,
        "test/sdr_i": 4.5,
        "test/si_snr": 11.1,
        "test/si_snr_i": 3.4,
    }
    update_final_metrics_with_test_summary(summary, str(exp_dir))

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["val_loss"] == 0.5
    assert payload["metrics"]["sdr"] == 12.3
    assert payload["metrics"]["sdr_i"] == 4.5
    assert payload["metrics"]["si_snr"] == 11.1
    assert payload["metrics"]["si_snr_i"] == 3.4
    assert payload["metrics"]["si_snri"] == 3.4
