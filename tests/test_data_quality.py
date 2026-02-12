"""Tests for data quality and validation."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from capacity_forecasting.data_generator import CapacityDataGenerator


@pytest.fixture
def sample_data():
    generator = CapacityDataGenerator(
        seed=42,
        n_samples=43200,
        n_cells=5,
        n_days=5,
        hours_per_day=24,
    )
    return generator.generate()


class TestDataQuality:
    def test_no_missing_values(self, sample_data):
        critical_cols = ["cell_id", "timestamp", "traffic_load_gb", "connected_users"]
        for col in critical_cols:
            if col in sample_data.columns:
                assert sample_data[col].isna().sum() == 0, f"Missing values in {col}"

    def test_data_types(self, sample_data):
        assert pd.api.types.is_datetime64_any_dtype(sample_data["timestamp"])
        assert pd.api.types.is_numeric_dtype(sample_data["traffic_load_gb"])

    def test_value_ranges(self, sample_data):
        assert sample_data["traffic_load_gb"].min() > 0
        assert sample_data["prb_utilization"].min() >= 0
        assert sample_data["prb_utilization"].max() <= 1
        assert sample_data["connected_users"].min() > 0

    def test_categorical_values(self, sample_data):
        assert set(sample_data["cell_type"].unique()).issubset({"macro", "micro", "small"})
        assert set(sample_data["area_type"].unique()).issubset({"urban", "suburban", "rural"})
        assert set(sample_data["day_type"].unique()).issubset({"weekday", "weekend"})

    def test_sample_size(self, sample_data):
        # 5 cells * 5 days * 24 hours = 600
        assert len(sample_data) == 600

    def test_growth_trend(self, sample_data):
        # Traffic should show general increase over time
        # Compare first 20% mean to last 20% mean
        n = len(sample_data)
        sorted_data = sample_data.sort_values("timestamp")
        first_segment = sorted_data.head(int(n * 0.2))
        last_segment = sorted_data.tail(int(n * 0.2))
        assert last_segment["traffic_load_gb"].mean() >= first_segment["traffic_load_gb"].mean(), (
            "Traffic should generally increase over time"
        )


class TestDataGenerator:
    def test_generator_reproducibility(self):
        gen1 = CapacityDataGenerator(
            seed=42,
            n_samples=43200,
            n_cells=5,
            n_days=5,
            hours_per_day=24,
        )
        gen2 = CapacityDataGenerator(
            seed=42,
            n_samples=43200,
            n_cells=5,
            n_days=5,
            hours_per_day=24,
        )
        df1 = gen1.generate()
        df2 = gen2.generate()
        pd.testing.assert_frame_equal(df1, df2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
