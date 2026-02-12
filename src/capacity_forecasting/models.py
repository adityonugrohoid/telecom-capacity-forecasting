"""
ML model training and evaluation for Telecom Capacity Forecasting.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from .config import MODEL_CONFIG


class BaseModel:
    """Base class for ML models."""

    def __init__(self, config: dict = None):
        self.config = config or MODEL_CONFIG
        self.model = None
        self.feature_names = None
        self.is_trained = False

    def prepare_data(self, df, target_col, test_size=0.2, random_state=42):
        y = df[target_col]
        X = df.drop(columns=[target_col])
        self.feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"Train set: {X_train.shape[0]:,} samples")
        print(f"Test set: {X_test.shape[0]:,} samples")
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        raise NotImplementedError("Subclasses must implement train()")

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, task_type="classification"):
        y_pred = self.predict(X_test)
        metrics = {}
        if task_type == "classification":
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            metrics["precision"] = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            metrics["recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            metrics["f1"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            if len(np.unique(y_test)) == 2:
                y_proba = self.model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        elif task_type == "regression":
            metrics["mse"] = mean_squared_error(y_test, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_test, y_pred)
            metrics["r2"] = 1 - (
                np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
            )
        return metrics

    def get_feature_importance(self):
        if not hasattr(self.model, "feature_importances_"):
            raise AttributeError("Model does not support feature importance")
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

    def save(self, filepath):
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


def cross_validate_model(model, X, y, cv_folds=5, scoring="accuracy"):
    """Perform cross-validation on a trained model."""
    scores = cross_val_score(model.model, X, y, cv=cv_folds, scoring=scoring)
    return {
        "mean_score": scores.mean(),
        "std_score": scores.std(),
        "all_scores": scores,
    }


def print_metrics(metrics, title="Model Performance"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:8.4f}")
    print(f"{'=' * 50}\n")


class LightGBMForecaster(BaseModel):
    """LightGBM-based regressor for capacity time-series forecasting."""

    def __init__(self, config: dict = None):
        super().__init__(config)
        import lightgbm as lgb

        self.lgb = lgb

    def prepare_time_series_data(self, df, target_col, test_ratio=0.2):
        """Prepare data using a chronological split instead of random split.

        The last ``test_ratio`` fraction of the data (by row order, which is
        assumed to be sorted by time) is used as the test set so that the model
        is always evaluated on future data.
        """
        y = df[target_col]
        X = df.drop(columns=[target_col])
        self.feature_names = X.columns.tolist()

        split_idx = int(len(df) * (1 - test_ratio))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"Train set: {X_train.shape[0]:,} samples (earliest {100 * (1 - test_ratio):.0f}%)")
        print(f"Test set:  {X_test.shape[0]:,} samples (latest {100 * test_ratio:.0f}%)")
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """Train the LightGBM regressor with time-series-appropriate params."""
        params = self.config.get("lgbm_params", {})
        self.model = self.lgb.LGBMRegressor(
            num_leaves=params.get("num_leaves", 63),
            learning_rate=params.get("learning_rate", 0.05),
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", -1),
            min_child_samples=params.get("min_child_samples", 30),
            subsample=params.get("subsample", 0.7),
            colsample_bytree=params.get("colsample_bytree", 0.7),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 0.5),
            random_state=params.get("random_state", 42),
            n_jobs=params.get("n_jobs", -1),
            verbose=params.get("verbose", -1),
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("LightGBM Capacity Forecaster trained successfully.")
        return self

    def evaluate(self, X_test, y_test, task_type="regression"):
        """Evaluate with regression metrics plus MAPE."""
        metrics = super().evaluate(X_test, y_test, task_type="regression")

        # Add Mean Absolute Percentage Error (MAPE)
        y_pred = self.predict(X_test)
        mask = y_test != 0
        if mask.any():
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        else:
            mape = float("inf")
        metrics["mape_pct"] = mape

        return metrics


class ProphetForecaster(BaseModel):
    """Simple seasonal-decomposition forecaster using statsmodels.

    This provides a lightweight alternative to Facebook Prophet by leveraging
    ``statsmodels.tsa.seasonal`` for trend/seasonal decomposition and
    extrapolating the trend with a simple linear model.
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        try:
            from sklearn.linear_model import LinearRegression
            from statsmodels.tsa.seasonal import seasonal_decompose

            self._seasonal_decompose = seasonal_decompose
            self._LinearRegression = LinearRegression
        except ImportError as exc:
            raise ImportError(
                "statsmodels is required for ProphetForecaster. "
                "Install it with: pip install statsmodels"
            ) from exc

        self._decomposition = None
        self._trend_model = None
        self._seasonal_cycle = None
        self._period = None

    def train(self, X_train, y_train):
        """Fit a seasonal decomposition + linear trend model.

        ``X_train`` is expected to be a DataFrame whose index (or a column
        named 'timestamp' / 'date') represents time.  ``y_train`` is the
        target series.
        """
        period = self.config.get("seasonal_period", 24)
        self._period = period

        # Decompose
        self._decomposition = self._seasonal_decompose(
            y_train, model="additive", period=period, extrapolate_trend="freq"
        )

        # Fit a linear trend
        trend = self._decomposition.trend.dropna()
        X_idx = np.arange(len(trend)).reshape(-1, 1)
        self._trend_model = self._LinearRegression()
        self._trend_model.fit(X_idx, trend.values)

        # Store one full seasonal cycle for repeating during prediction
        seasonal = self._decomposition.seasonal
        self._seasonal_cycle = seasonal.values[:period]

        self.is_trained = True
        print("ProphetForecaster (statsmodels decomposition) trained successfully.")
        return self

    def predict(self, X):
        """Forecast future values based on decomposition.

        ``X`` can be a DataFrame or an integer indicating how many steps ahead
        to forecast.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if isinstance(X, (int, np.integer)):
            n_steps = int(X)
        else:
            n_steps = len(X)

        # Extrapolate trend
        trend_start = len(self._decomposition.trend)
        future_idx = np.arange(trend_start, trend_start + n_steps).reshape(-1, 1)
        trend_pred = self._trend_model.predict(future_idx)

        # Tile seasonal component
        full_seasonal = np.tile(self._seasonal_cycle, (n_steps // self._period) + 1)[:n_steps]

        return trend_pred + full_seasonal


def main():
    """Main entry point for capacity forecasting model training."""
    # TODO: Load processed data from PROCESSED_DATA_DIR
    # TODO: Instantiate LightGBMForecaster or ProphetForecaster
    # TODO: Prepare time-series data, train, evaluate, and save model
    print("Capacity Forecasting model training — not yet implemented.")


if __name__ == "__main__":
    main()
