# Telecom Capacity Forecasting

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)
![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

## Business Context

Under-provisioned cells degrade QoE and drive churn. Over-provisioning wastes CAPEX. Accurate traffic forecasting enables just-in-time capacity upgrades, optimizing both customer experience and infrastructure investment.

## Problem Framing

Time-series forecasting using LightGBM + Prophet/statsmodels.

- **Target:** `traffic_load_gb`
- **Primary Metric:** MAPE
- **Challenges:**
  - Multiple seasonalities (24h diurnal + 7d weekly)
  - Growth trends (~2%/month)
  - Special events with 2-3x traffic multiplier
  - Per-cell variation in traffic patterns

## Data Engineering

Hourly cell-level time-series (60 cells x 30 days x 24h = 43.2K rows):

- **24h diurnal pattern** -- realistic hourly load curves with morning ramp, afternoon peak, and overnight trough
- **Weekly seasonality** -- weekday/weekend traffic shape differences
- **Growth trend** -- 2%/month linear growth overlaid on seasonal patterns
- **Special events** -- ~2% of hours flagged as events with 2-3x traffic multiplier

Domain physics: traffic follows predictable diurnal and weekly cycles driven by subscriber behavior. Special events (concerts, sports, holidays) create localized surges that deviate from baseline patterns.

## Methodology

- LightGBM with temporal lag features and rolling aggregates
- **Feature groups:**
  - Lag features (1h, 24h, 168h/7d lookback)
  - Rolling aggregates (mean, std over 24h and 7d windows)
  - Calendar features (hour-of-day, day-of-week, is_weekend)
  - Growth trend component
- Chronological train/test split (no shuffling to preserve temporal order)
- Optional Prophet for seasonal decomposition and trend extraction

## Key Findings

- **MAPE:** 14.5%, **R²:** 0.90 on held-out test set (chronological split, no contemporaneous features)
- **Top predictors:** lag features (1h, 24h, 168h) and lagged rolling aggregates dominate SHAP importance
- Peak-hour MAPE (15.0%) and off-peak MAPE (14.3%) are well-balanced, indicating consistent accuracy across traffic regimes
- Model uses only historically available features (lags, rolling stats, calendar) -- no contemporaneous KPIs that would constitute data leakage in a real forecasting scenario

## Quick Start

```bash
# Clone the repository
git clone https://github.com/adityonugrohoid/telecom-ml-portfolio.git
cd telecom-ml-portfolio/05-capacity-forecasting

# Install dependencies
uv sync

# Generate synthetic data
uv run python -m capacity_forecasting.data_generator

# Run the notebook
uv run jupyter lab notebooks/
```

## Project Structure

```
05-capacity-forecasting/
├── README.md
├── pyproject.toml
├── notebooks/
│   └── 05_capacity_forecasting.ipynb
├── src/
│   └── capacity_forecasting/
│       ├── __init__.py
│       ├── config.py
│       ├── data_generator.py
│       ├── features.py
│       └── models.py
├── data/
│   └── .gitkeep
├── tests/
│   └── test_data_quality.py
└── docs/
```

## Related Projects

| # | Project | Description |
|---|---------|-------------|
| 1 | [Churn Prediction](../01-churn-prediction) | Binary classification to predict customer churn |
| 2 | [Root Cause Analysis](../02-root-cause-analysis) | Multi-class classification for network alarm RCA |
| 3 | [Anomaly Detection](../03-anomaly-detection) | Unsupervised detection of network anomalies |
| 4 | [QoE Prediction](../04-qoe-prediction) | Regression to predict quality of experience |
| 5 | **Capacity Forecasting** (this repo) | Time-series forecasting for network capacity planning |
| 6 | [Network Optimization](../06-network-optimization) | Optimization of network resource allocation |

## License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

## Author

**Adityo Nugroho**
