"""Baseline comparison for Telecom Capacity Forecasting.

The README names ProphetForecaster (a statsmodels seasonal decomposition
plus linear trend extrapolation) as the baseline for the LightGBM
forecaster, but that baseline has never actually been scored. This module
generates data with the project's own generator, runs it through the
project's own feature pipeline, makes one chronological split, and fits
both models on that split so the two can be compared on the same numbers.
"""

import json

import numpy as np

from .config import DATA_GEN_CONFIG, PROJECT_ROOT
from .data_generator import CapacityDataGenerator
from .features import FeatureEngineer
from .models import LightGBMForecaster, ProphetForecaster

TARGET = "traffic_load_gb"

# Columns dropped before modelling: identifiers, plus contemporaneous KPIs and
# the interaction features derived from them. These are measured at the same
# timestamp as the target and would leak the answer into a forecast; this
# list matches notebooks/05_capacity_forecasting.ipynb.
DROP_COLS = [
    "cell_id",
    "timestamp",
    "connected_users",
    "prb_utilization",
    "avg_throughput_mbps",
    "avg_latency_ms",
    "avg_sinr_db",
    "load_per_user",
    "utilization_gap",
    "congestion_proxy",
    "throughput_efficiency",
]

EVIDENCE_PATH = PROJECT_ROOT / "evidence" / "baseline_metrics.json"


def _mape_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, matching LightGBMForecaster.evaluate.

    BaseModel.evaluate does not compute MAPE for its regression branch, so
    ProphetForecaster (a plain BaseModel subclass) never gets a MAPE score
    from the project's own evaluate() method. This repeats the exact
    calculation LightGBMForecaster.evaluate adds for its own metrics, so the
    baseline and the model are compared on the same measure.
    """
    mask = y_true != 0
    if not mask.any():
        raise ValueError("Cannot compute MAPE: every y_true value is zero")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def run_baseline_comparison(seed: int, n_cells: int, n_days: int, hours_per_day: int) -> dict:
    """Score the LightGBM forecaster against the README's named baseline.

    Generates data with CapacityDataGenerator, engineers features with
    FeatureEngineer, makes one split with
    LightGBMForecaster.prepare_time_series_data, and fits both
    LightGBMForecaster and ProphetForecaster on that same split.
    """
    n_samples = n_cells * n_days * hours_per_day
    raw = CapacityDataGenerator(
        seed=seed,
        n_samples=n_samples,
        n_cells=n_cells,
        n_days=n_days,
        hours_per_day=hours_per_day,
    ).generate()
    raw = raw.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)

    engineered = FeatureEngineer().pipeline(raw)
    drop_cols = [c for c in DROP_COLS if c in engineered.columns]
    model_df = engineered.drop(columns=drop_cols)

    forecaster = LightGBMForecaster()
    X_train, X_test, y_train, y_test = forecaster.prepare_time_series_data(
        model_df, target_col=TARGET, test_ratio=0.2
    )

    forecaster.train(X_train, y_train)
    model_metrics = forecaster.evaluate(X_test, y_test)

    baseline = ProphetForecaster()
    baseline.train(X_train, y_train)
    baseline_metrics = baseline.evaluate(X_test, y_test, task_type="regression")
    baseline_metrics["mape_pct"] = _mape_pct(y_test.to_numpy(), baseline.predict(X_test))

    return {
        "seed": seed,
        "n_cells": n_cells,
        "n_days": n_days,
        "hours_per_day": hours_per_day,
        "n_samples": n_samples,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "model": model_metrics,
        "baseline": baseline_metrics,
    }


def main() -> None:
    """Run the comparison at the README's documented scale and seed."""
    params = DATA_GEN_CONFIG["use_case_params"]
    results = run_baseline_comparison(
        seed=DATA_GEN_CONFIG["random_seed"],
        n_cells=params["n_cells"],
        n_days=params["n_days"],
        hours_per_day=params["hours_per_day"],
    )
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVIDENCE_PATH, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"Wrote baseline comparison to {EVIDENCE_PATH}")


if __name__ == "__main__":
    main()
