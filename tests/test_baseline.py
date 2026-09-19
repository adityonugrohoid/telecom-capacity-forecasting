"""Tests for the baseline comparison against the README's named baseline."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from capacity_forecasting.baseline import run_baseline_comparison

EXPECTED_METRIC_KEYS = {"mse", "rmse", "mae", "r2", "mape_pct"}


@pytest.fixture(scope="module")
def comparison():
    return run_baseline_comparison(seed=42, n_cells=10, n_days=30, hours_per_day=24)


class TestBaselineComparison:
    def test_returns_both_blocks_with_expected_keys(self, comparison):
        assert {"model", "baseline", "seed", "n_cells", "n_days", "train_rows", "test_rows"} <= set(
            comparison
        )
        assert set(comparison["model"]) == EXPECTED_METRIC_KEYS
        assert set(comparison["baseline"]) == EXPECTED_METRIC_KEYS

    def test_model_beats_the_baseline_on_mape(self, comparison):
        assert comparison["model"]["mape_pct"] <= comparison["baseline"]["mape_pct"]
