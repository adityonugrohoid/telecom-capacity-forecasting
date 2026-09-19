"""Tests for model training and evaluation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from capacity_forecasting.data_generator import CapacityDataGenerator
from capacity_forecasting.features import FeatureEngineer
from capacity_forecasting.models import BaseModel, LightGBMForecaster

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

# The README reports MAPE 14.5% on the full 43,200-row, 60-cell dataset. This
# test trains on a 10-cell subsample to stay fast, so a higher error is
# expected; the measured value at seed=42 is 20.9%. The ceiling below leaves
# a wide margin above that so it only trips on an actual regression in the
# generator or feature pipeline, not on ordinary variation.
MAPE_CEILING = 30.0

# LightGBMForecaster.prepare_time_series_data only produces a genuine
# chronological split when its input is a single, continuously-timestamped
# series. The pipeline output for the project's real multi-cell dataset is
# ordered by (cell_id, timestamp), not by timestamp alone, so a plain
# position-based split there ends up holding out entire cells rather than
# a time window. That is reported separately; this fixture uses one cell so
# the split's documented precondition actually holds.
N_CELLS_SPLIT_CHECK = 1


def _engineer(raw):
    """Run the project's feature pipeline the way the notebook does."""
    raw = raw.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)
    engineered = FeatureEngineer().pipeline(raw)
    drop_cols = [c for c in DROP_COLS if c in engineered.columns]
    return engineered, engineered.drop(columns=drop_cols)


@pytest.fixture(scope="module")
def model_df():
    raw = CapacityDataGenerator(
        seed=42, n_samples=10 * 30 * 24, n_cells=10, n_days=30, hours_per_day=24
    ).generate()
    _, df = _engineer(raw)
    return df


@pytest.fixture(scope="module")
def split(model_df):
    return LightGBMForecaster().prepare_time_series_data(
        model_df, target_col=TARGET, test_ratio=0.2
    )


@pytest.fixture(scope="module")
def trained(split):
    """The model as the project uses it: chronological split, then train."""
    X_train, _, y_train, _ = split
    model = LightGBMForecaster()
    model.train(X_train, y_train)
    return model


class TestTraining:
    def test_untrained_model_refuses_to_predict(self, split):
        _, X_test, _, _ = split
        with pytest.raises(ValueError):
            LightGBMForecaster().predict(X_test)

    def test_base_model_has_no_training(self, split):
        X_train, _, y_train, _ = split
        with pytest.raises(NotImplementedError):
            BaseModel().train(X_train, y_train)

    def test_training_marks_the_model_trained(self, trained):
        assert trained.is_trained

    def test_training_is_reproducible(self, split, trained):
        X_train, X_test, y_train, _ = split
        again = LightGBMForecaster()
        again.train(X_train, y_train)
        np.testing.assert_array_equal(trained.predict(X_test), again.predict(X_test))


class TestForecast:
    def test_forecast_shape_and_non_negative_load(self, split, trained):
        _, X_test, _, _ = split
        preds = trained.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert (preds >= 0).all()

    def test_metrics_are_complete(self, split, trained):
        _, X_test, _, y_test = split
        metrics = trained.evaluate(X_test, y_test)
        assert set(metrics) == {"mse", "rmse", "mae", "r2", "mape_pct"}

    def test_headline_mape_stays_below_ceiling(self, split, trained):
        _, X_test, _, y_test = split
        metrics = trained.evaluate(X_test, y_test)
        assert metrics["mape_pct"] < MAPE_CEILING, f"MAPE rose to {metrics['mape_pct']:.2f}%"


class TestChronologicalSplit:
    def test_split_keeps_train_before_test_in_time(self):
        raw = CapacityDataGenerator(
            seed=42,
            n_samples=N_CELLS_SPLIT_CHECK * 30 * 24,
            n_cells=N_CELLS_SPLIT_CHECK,
            n_days=30,
            hours_per_day=24,
        ).generate()
        engineered, single_cell_df = _engineer(raw)
        X_train, X_test, _, _ = LightGBMForecaster().prepare_time_series_data(
            single_cell_df, target_col=TARGET, test_ratio=0.2
        )
        train_timestamps = engineered.loc[X_train.index, "timestamp"]
        test_timestamps = engineered.loc[X_test.index, "timestamp"]
        assert train_timestamps.max() < test_timestamps.min()


class TestPersistence:
    def test_saved_model_predicts_the_same(self, split, trained, tmp_path):
        _, X_test, _, _ = split
        path = tmp_path / "capacity.pkl"
        trained.save(path)
        restored = LightGBMForecaster()
        restored.load(path)
        np.testing.assert_array_equal(trained.predict(X_test), restored.predict(X_test))

    def test_untrained_model_refuses_to_save(self, tmp_path):
        with pytest.raises(ValueError):
            LightGBMForecaster().save(tmp_path / "capacity.pkl")
