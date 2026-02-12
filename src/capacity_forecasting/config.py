"""
Configuration management for Telecom Capacity Forecasting.

This module centralizes all configuration parameters, making it easy to
adjust settings without modifying core logic.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


# ============================================================================
# DATA GENERATION CONFIG
# ============================================================================

DATA_GEN_CONFIG = {
    "random_seed": 42,
    "n_samples": 43_200,  # 60 cells x 30 days x 24 hours
    "test_size": 0.2,
    "validation_size": 0.1,
    "use_case_params": {
        "n_cells": 60,
        "n_days": 30,
        "hours_per_day": 24,
        "growth_rate_monthly": 0.02,
        "special_event_probability": 0.02,
        "special_event_multiplier": 2.5,
    },
}


# ============================================================================
# FEATURE ENGINEERING CONFIG
# ============================================================================

FEATURE_CONFIG = {
    "categorical_features": [
        "cell_type",
        "area_type",
        "day_type",
    ],
    "numerical_features": [
        "traffic_load_gb",
        "connected_users",
        "prb_utilization",
        "avg_throughput_mbps",
        "avg_latency_ms",
    ],
    "datetime_features": ["timestamp"],
    "rolling_windows": [24, 168],  # 1 day, 1 week in hours
    "lag_features": [1, 24, 168],  # 1h, 24h, 7d lags
    "create_features": True,
}


# ============================================================================
# MODEL TRAINING CONFIG
# ============================================================================

MODEL_CONFIG = {
    "algorithm": "lightgbm",
    "cv_folds": 5,
    "cv_strategy": "time_series",
    "hyperparameters": {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    },
    "early_stopping_rounds": 10,
    "verbose": True,
}


# ============================================================================
# EVALUATION CONFIG
# ============================================================================

EVAL_CONFIG = {
    "primary_metric": "mape",
    "threshold": None,
    "compute_metrics": [
        "mape",
        "rmse",
        "mae",
        "r2",
    ],
}


# ============================================================================
# VISUALIZATION CONFIG
# ============================================================================

VIZ_CONFIG = {
    "style": "whitegrid",
    "palette": "husl",
    "context": "notebook",
    "figure_size": (12, 6),
    "dpi": 100,
}


# ============================================================================
# UTILITIES
# ============================================================================


def ensure_directories() -> None:
    """Create necessary directories if they don't exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_custom_config(config_path: Path) -> Dict[str, Any]:
    """Load custom configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    return {
        "data_gen": DATA_GEN_CONFIG,
        "features": FEATURE_CONFIG,
        "model": MODEL_CONFIG,
        "eval": EVAL_CONFIG,
        "viz": VIZ_CONFIG,
        "paths": {
            "root": PROJECT_ROOT,
            "data": DATA_DIR,
            "raw": RAW_DATA_DIR,
            "processed": PROCESSED_DATA_DIR,
            "notebooks": NOTEBOOKS_DIR,
        },
    }


if __name__ == "__main__":
    ensure_directories()
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Data directory: {DATA_DIR}")
    print(f"Random seed: {DATA_GEN_CONFIG['random_seed']}")
    print(f"Algorithm: {MODEL_CONFIG['algorithm']}")
