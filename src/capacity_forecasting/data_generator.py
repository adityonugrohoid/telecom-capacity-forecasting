"""
Domain-informed synthetic data generator for Telecom Capacity Forecasting.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_GEN_CONFIG, RAW_DATA_DIR, ensure_directories


class TelecomDataGenerator:
    """Base class for generating synthetic telecom data."""

    def __init__(self, seed: int = 42, n_samples: int = 10_000):
        self.seed = seed
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def generate(self) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement generate()")

    def generate_sinr(
        self, n: int, base_sinr_db: float = 10.0, noise_std: float = 5.0
    ) -> np.ndarray:
        sinr = self.rng.normal(base_sinr_db, noise_std, n)
        return np.clip(sinr, -5, 25)

    def sinr_to_throughput(
        self, sinr_db: np.ndarray, network_type: np.ndarray, noise_factor: float = 0.2
    ) -> np.ndarray:
        sinr_linear = 10 ** (sinr_db / 10)
        capacity_factor = np.log2(1 + sinr_linear)
        max_throughput = np.where(network_type == "5G", 300, 50)
        throughput = capacity_factor * max_throughput / 5
        noise = self.rng.normal(1, noise_factor, len(throughput))
        throughput = throughput * noise
        return np.clip(throughput, 0.1, max_throughput)

    def generate_congestion_pattern(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        hour = timestamps.hour
        day_of_week = timestamps.dayofweek
        congestion = 0.5 + 0.3 * np.sin((hour - 6) * np.pi / 12)
        peak_morning = (hour >= 9) & (hour <= 11)
        peak_evening = (hour >= 18) & (hour <= 21)
        congestion = np.where(peak_morning | peak_evening, congestion * 1.3, congestion)
        is_weekend = day_of_week >= 5
        congestion = np.where(is_weekend, congestion * 0.8, congestion)
        noise = self.rng.normal(0, 0.1, len(congestion))
        congestion = congestion + noise
        return np.clip(congestion, 0, 1)

    def congestion_to_latency(
        self, congestion: np.ndarray, base_latency_ms: float = 20
    ) -> np.ndarray:
        latency = base_latency_ms * (1 + 5 * congestion**2)
        jitter = self.rng.normal(0, 5, len(latency))
        latency = latency + jitter
        return np.clip(latency, 10, 300)

    def compute_qoe_mos(
        self,
        throughput_mbps: np.ndarray,
        latency_ms: np.ndarray,
        packet_loss_pct: np.ndarray,
        app_type: np.ndarray,
    ) -> np.ndarray:
        mos_throughput = 1 + 4 * (1 - np.exp(-throughput_mbps / 10))
        latency_penalty = np.clip(latency_ms / 100, 0, 2)
        loss_penalty = packet_loss_pct / 2
        mos = mos_throughput - latency_penalty - loss_penalty
        video_mask = app_type == "video_streaming"
        mos = np.where(video_mask, mos - packet_loss_pct * 0.5, mos)
        gaming_mask = app_type == "gaming"
        mos = np.where(gaming_mask, mos - latency_penalty * 0.5, mos)
        return np.clip(mos, 1, 5)

    def save(self, df: pd.DataFrame, filename: str) -> Path:
        ensure_directories()
        output_path = RAW_DATA_DIR / f"{filename}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df):,} rows to {output_path}")
        return output_path


class CapacityDataGenerator(TelecomDataGenerator):
    """Generate synthetic hourly cell-level time-series data for capacity forecasting."""

    def __init__(
        self,
        seed: int = 42,
        n_samples: int = 43_200,
        n_cells: int = 60,
        n_days: int = 30,
        hours_per_day: int = 24,
        growth_rate_monthly: float = 0.02,
        special_event_probability: float = 0.02,
        special_event_multiplier: float = 2.5,
    ):
        super().__init__(seed=seed, n_samples=n_samples)
        self.n_cells = n_cells
        self.n_days = n_days
        self.hours_per_day = hours_per_day
        self.growth_rate_monthly = growth_rate_monthly
        self.special_event_probability = special_event_probability
        self.special_event_multiplier = special_event_multiplier

    # ------------------------------------------------------------------
    # Cell topology helpers
    # ------------------------------------------------------------------

    def _create_cell_profiles(self) -> pd.DataFrame:
        """Create static cell metadata (type, area, base load, base users)."""
        cell_types = self.rng.choice(
            ["macro", "micro", "small"],
            size=self.n_cells,
            p=[0.4, 0.35, 0.25],
        )
        area_types = self.rng.choice(
            ["urban", "suburban", "rural"],
            size=self.n_cells,
            p=[0.5, 0.3, 0.2],
        )

        base_load_map = {"urban": 15.0, "suburban": 8.0, "rural": 3.0}
        base_users_map = {"urban": 500, "suburban": 300, "rural": 100}

        profiles = pd.DataFrame(
            {
                "cell_id": [f"CELL_{i:04d}" for i in range(self.n_cells)],
                "cell_type": cell_types,
                "area_type": area_types,
            }
        )
        profiles["base_load_gb"] = profiles["area_type"].map(base_load_map)
        profiles["base_users"] = profiles["area_type"].map(base_users_map)
        return profiles

    # ------------------------------------------------------------------
    # Seasonality & trend helpers
    # ------------------------------------------------------------------

    def _diurnal_factor(self, hour: np.ndarray) -> np.ndarray:
        """24-hour sinusoidal pattern peaking at hour 20, trough at hour 4."""
        return 1.0 + 0.5 * np.sin((hour - 4) * np.pi / 8 - np.pi / 2)

    def _weekly_factor(self, day_of_week: np.ndarray) -> np.ndarray:
        """Slightly lower traffic on weekends (Sat=5, Sun=6)."""
        is_weekend = day_of_week >= 5
        return np.where(is_weekend, 0.85, 1.0)

    def _growth_trend(self, day_index: np.ndarray) -> np.ndarray:
        """Linear 2 %/month growth over the observation window."""
        daily_rate = self.growth_rate_monthly / 30.0
        return 1.0 + daily_rate * day_index

    # ------------------------------------------------------------------
    # Main generation
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Generate the full hourly capacity dataset."""
        cell_profiles = self._create_cell_profiles()

        n_timesteps = self.n_days * self.hours_per_day
        start = pd.Timestamp("2024-01-01")
        timestamps = pd.date_range(start=start, periods=n_timesteps, freq="h")

        records = []

        for _, cell in cell_profiles.iterrows():
            cell_id = cell["cell_id"]
            cell_type = cell["cell_type"]
            area_type = cell["area_type"]
            base_load = cell["base_load_gb"]
            base_users = cell["base_users"]

            hour = timestamps.hour.values.astype(float)
            day_of_week = timestamps.dayofweek.values
            day_index = np.arange(n_timesteps) / self.hours_per_day

            # ----- traffic_load_gb (TARGET) -----
            diurnal = self._diurnal_factor(hour)
            weekly = self._weekly_factor(day_of_week)
            trend = self._growth_trend(day_index)

            # Base noise: moderate variance for realistic forecasting error
            noise = self.rng.normal(1.0, 0.12, n_timesteps)

            # Auto-correlated noise (AR(1) process): consecutive hours
            # share noise, so lag-1 features can't fully eliminate it.
            ar_noise = np.zeros(n_timesteps)
            ar_noise[0] = self.rng.normal(0, 0.04)
            for t in range(1, n_timesteps):
                ar_noise[t] = 0.6 * ar_noise[t - 1] + self.rng.normal(0, 0.04)

            # Cell-specific random walk drift (non-linear growth)
            # Some cells gain/lose subscribers unpredictably.
            drift = np.zeros(n_timesteps)
            drift[0] = 0
            for t in range(1, n_timesteps):
                drift[t] = drift[t - 1] + self.rng.normal(0, 0.002)
            drift_factor = 1.0 + np.clip(drift, -0.10, 0.10)

            traffic_load = base_load * diurnal * weekly * trend * noise * drift_factor
            traffic_load = traffic_load + ar_noise * base_load

            # Special events: multi-hour blocks (3-6 hours) with moderate
            # traffic spikes, harder to predict but not overwhelming.
            is_event = np.zeros(n_timesteps, dtype=bool)
            t = 0
            while t < n_timesteps:
                if self.rng.random() < self.special_event_probability * 0.5:
                    duration = int(self.rng.integers(3, 7))
                    is_event[t : min(t + duration, n_timesteps)] = True
                    t += duration
                else:
                    t += 1
            event_multiplier = self.rng.uniform(
                1.3, self.special_event_multiplier * 0.8, n_timesteps
            )
            traffic_load = np.where(is_event, traffic_load * event_multiplier, traffic_load)

            # Occasional cell outages (~0.1% of hours): traffic drops sharply
            outage_mask = self.rng.random(n_timesteps) < 0.001
            traffic_load = np.where(
                outage_mask, traffic_load * self.rng.uniform(0.05, 0.2), traffic_load
            )

            traffic_load = np.clip(traffic_load, 0.0, None)

            # ----- correlated KPIs -----
            # Congestion proxy (0-1) derived from how close load is to a cell max
            cell_max_load = base_load * 3.0
            congestion = np.clip(traffic_load / cell_max_load, 0.0, 1.0)

            connected_users = base_users * congestion + self.rng.normal(
                0, base_users * 0.05, n_timesteps
            )
            connected_users = np.clip(connected_users, 10, base_users * 1.5).astype(int)

            prb_utilization = 0.1 + 0.85 * congestion + self.rng.normal(0, 0.03, n_timesteps)
            prb_utilization = np.clip(prb_utilization, 0.1, 0.95)

            # Throughput: inversely related to congestion
            max_tp = 100.0 if area_type == "urban" else (60.0 if area_type == "suburban" else 30.0)
            avg_throughput = max_tp * (1.0 - 0.7 * congestion) + self.rng.normal(0, 3, n_timesteps)
            avg_throughput = np.clip(avg_throughput, 1.0, max_tp)

            # Latency: positively related to congestion
            avg_latency = self.congestion_to_latency(congestion)

            # SINR: use base-class helper; area-dependent base
            sinr_base = {"urban": 8.0, "suburban": 12.0, "rural": 15.0}[area_type]
            avg_sinr = self.generate_sinr(n_timesteps, base_sinr_db=sinr_base, noise_std=4.0)

            day_type = np.where(day_of_week >= 5, "weekend", "weekday")

            cell_df = pd.DataFrame(
                {
                    "cell_id": cell_id,
                    "cell_type": cell_type,
                    "area_type": area_type,
                    "timestamp": timestamps,
                    "traffic_load_gb": np.round(traffic_load, 4),
                    "connected_users": connected_users,
                    "prb_utilization": np.round(prb_utilization, 4),
                    "avg_throughput_mbps": np.round(avg_throughput, 2),
                    "avg_latency_ms": np.round(avg_latency, 2),
                    "avg_sinr_db": np.round(avg_sinr, 2),
                    "day_type": day_type,
                }
            )
            records.append(cell_df)

        df = pd.concat(records, ignore_index=True)
        return df


def main() -> None:
    """Generate and save the capacity-forecasting dataset using project config."""
    config = DATA_GEN_CONFIG
    params = config["use_case_params"]

    generator = CapacityDataGenerator(
        seed=config["random_seed"],
        n_samples=config["n_samples"],
        n_cells=params["n_cells"],
        n_days=params["n_days"],
        hours_per_day=params["hours_per_day"],
        growth_rate_monthly=params["growth_rate_monthly"],
        special_event_probability=params["special_event_probability"],
        special_event_multiplier=params["special_event_multiplier"],
    )

    df = generator.generate()

    print(f"Generated dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Unique cells: {df['cell_id'].nunique()}")
    print("\nSample statistics for traffic_load_gb (target):")
    print(df["traffic_load_gb"].describe())

    generator.save(df, "capacity_forecasting")


if __name__ == "__main__":
    main()
