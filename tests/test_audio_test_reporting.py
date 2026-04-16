import json
import sys
from pathlib import Path
from pathlib import Path
from unittest.mock import MagicMock, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_test import (
    build_test_summary,
    load_wandb_run_metadata,
    log_test_summary_to_wandb,
    print_test_summary,
)


def test_build_test_summary_prefixes_metrics_with_test_namespace():
    metrics_row = {
        "snt_id": "avg",
        "sdr": "1.25",
        "sdr_i": "2.5",
        "si-snr": "3.75",
        "si-snr_i": "4.0",
    }

    summary = build_test_summary(metrics_row)

    assert summary == {
        "test/sdr": 1.25,
        "test/sdr_i": 2.5,
        "test/si_snr": 3.75,
        "test/si_snr_i": 4.0,
    }


def test_print_test_summary_emits_readable_lines():
    summary = {
        "test/sdr": 1.25,
        "test/sdr_i": 2.5,
        "test/si_snr": 3.75,
        "test/si_snr_i": 4.0,
    }
    printer = Mock()

    print_test_summary(summary, print_fn=printer)

    printer.assert_any_call("Test Summary")
    printer.assert_any_call("test/sdr: 1.250")
    printer.assert_any_call("test/sdr_i: 2.500")
    printer.assert_any_call("test/si_snr: 3.750")
    printer.assert_any_call("test/si_snr_i: 4.000")


def test_load_wandb_run_metadata_reads_saved_training_run_metadata(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    metadata_path = exp_dir / "wandb_run.json"
    metadata = {"entity": "demo-entity", "project": "demo-project", "run_id": "abc123"}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    loaded = load_wandb_run_metadata(str(exp_dir))

    assert loaded == metadata


def test_log_test_summary_to_wandb_updates_existing_training_run_summary(tmp_path):
    summary = {
        "test/sdr": 1.25,
        "test/sdr_i": 2.5,
    }
    run = Mock()
    api = Mock()
    api.run.return_value = run
    wandb_module = Mock()
    wandb_module.Api.return_value = api
    train_conf = {
        "exp": {"exp_name": "TIGER-MiniLibriMix"},
        "audionet": {"audionet_name": "TIGER"},
        "datamodule": {
            "data_name": "Libri2MixModuleRemix",
            "data_config": {"batch_size": 1, "segment": 1.0},
        },
    }
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "wandb_run.json").write_text(
        json.dumps(
            {"entity": "demo-entity", "project": "demo-project", "run_id": "abc123"}
        ),
        encoding="utf-8",
    )

    log_test_summary_to_wandb(summary, train_conf, str(exp_dir), wandb_module=wandb_module)

    api.run.assert_called_once_with("demo-entity/demo-project/abc123")
    run.summary.update.assert_called_once_with(summary)
