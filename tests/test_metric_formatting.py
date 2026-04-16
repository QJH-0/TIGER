import sys
from pathlib import Path
from unittest.mock import Mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from look2hear.metrics.wrapper import format_metric_value
from look2hear.metrics.splitwrapper import SPlitMetricsTracker


def test_format_metric_value_limits_decimal_places_without_padding():
    assert format_metric_value(1.23456) == "1.235"
    assert format_metric_value(1.2) == "1.2"
    assert format_metric_value(1.0) == "1"
    assert format_metric_value(-0.98765) == "-0.988"


def test_split_metrics_tracker_does_not_write_per_sample_rows():
    tracker = SPlitMetricsTracker.__new__(SPlitMetricsTracker)
    tracker.one_all_snrs = []
    tracker.one_all_snrs_i = []
    tracker.one_all_sisnrs = []
    tracker.one_all_sisnrs_i = []
    tracker.two_all_snrs = []
    tracker.two_all_snrs_i = []
    tracker.two_all_sisnrs = []
    tracker.two_all_sisnrs_i = []
    tracker.writer = Mock()
    tracker.pit_sisnr = lambda *args, **kwargs: torch.tensor(-1.23456)

    def fake_pit_snr(*args, **kwargs):
        if kwargs.get("return_ests"):
            return torch.tensor(0.0), args[0]
        return torch.tensor(-2.34567)

    tracker.pit_snr = fake_pit_snr

    mix = torch.tensor([0.1, 0.2, 0.3])
    clean = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
        ]
    )
    estimate = clean.clone()

    tracker(mix=mix, clean=clean, estimate=estimate, key="sample")

    tracker.writer.writerow.assert_not_called()
