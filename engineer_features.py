"""
feature_pipeline/engineer_features.py

This module builds machine learning features from raw AQI data.
It includes:
- Time-based features
- Lag/rolling features (historical AQI trends)
- Derived environmental interactions
- Future AQI prediction targets
"""

import numpy as np
import pandas as pd
from pathlib import Path


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features and convert cyclic time into sin/cos form."""
    df = df.copy()

    # Convert timestamp column to datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Extract basic time components
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek  # 0 = Monday
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # Cyclical encoding to preserve time continuity
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_lag_features(df: pd.DataFrame, group_col: str = "city") -> pd.DataFrame:
    """Add historical AQI-based lag and rolling statistics per city."""
    df = df.sort_values(["city", "timestamp"]).copy()

    # Create lag features (past AQI values)
    for lag in [1, 3, 6, 12, 24]:
        df[f"aqi_lag_{lag}h"] = df.groupby(group_col)["aqi"].shift(lag)

    # Rolling mean over short and long time windows
    df["aqi_rolling_mean_6h"] = df.groupby(group_col)["aqi"].transform(
        lambda x: x.shift(1).rolling(6, min_periods=1).mean()
    )

    df["aqi_rolling_mean_24h"] = df.groupby(group_col)["aqi"].transform(
        lambda x: x.shift(1).rolling(24, min_periods=1).mean()
    )

    # Rolling volatility (standard deviation)
    df["aqi_rolling_std_6h"] = df.groupby(group_col)["aqi"].transform(
        lambda x: x.shift(1).rolling(6, min_periods=1).std().fillna(0)
    )

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction-based and domain-specific environmental features."""
    df = df.copy()

    # Short-term AQI change (trend/momentum)
    df["aqi_change_rate"] = df["aqi"] - df.get("aqi_lag_3h", df["aqi"])

    # Ratio between PM2.5 and PM10
    df["pm_ratio"] = df["pm25"] / (df["pm10"] + 1e-6)

    # Interaction: wind affects pollution dispersion
    df["wind_humidity"] = df["wind_speed"] * df["humidity"] / 100

    # Heat index approximation (temperature + humidity effect)
    df["heat_index"] = (
        df["temperature"] +
        0.33 * (df["humidity"] / 100 * 6.105) -
        4
    )

    return df


def compute_targets(df: pd.DataFrame, group_col: str = "city") -> pd.DataFrame:
    """
    Create prediction targets for future AQI values.
    Model learns to predict AQI 24h, 48h, and 72h ahead.
    """
    df = df.sort_values(["city", "timestamp"]).copy()

    # Shift AQI into future values (forecasting target)
    for horizon in [24, 48, 72]:
        df[f"target_aqi_{horizon}h"] = df.groupby(group_col)["aqi"].shift(-horizon)

    return df


# -------------------------------
# Feature and target definitions
# -------------------------------

FEATURE_COLS = [
    "pm25", "pm10", "o3", "no2", "co", "so2",
    "temperature", "humidity", "pressure", "wind_speed", "wind_deg", "visibility",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h",
    "aqi_rolling_mean_6h", "aqi_rolling_mean_24h", "aqi_rolling_std_6h",
    "aqi_change_rate", "pm_ratio", "wind_humidity", "heat_index",
]

TARGET_COLS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]


def run_pipeline(input_path: str = "data/raw_backfill.csv",
                 output_path: str = "data/features.csv") -> pd.DataFrame:
    """Full feature engineering pipeline: raw data → ML features dataset."""

    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)

    print(f"  Rows: {len(df)}, Cities: {df['city'].nunique()}")

    print("Adding time features...")
    df = add_time_features(df)

    print("Adding lag features...")
    df = add_lag_features(df)

    print("Adding derived features...")
    df = add_derived_features(df)

    print("Computing targets...")
    df = compute_targets(df)

    # Remove rows with missing lag/target values (edge effects)
    df = df.dropna(subset=TARGET_COLS + ["aqi_lag_24h"])

    print(f"  Final rows after dropping NaN: {len(df)}")

    # Select only valid feature columns
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    output_df = df[["city", "timestamp"] + available_features + TARGET_COLS]

    # Save processed dataset
    output_df.to_csv(output_path, index=False)

    print(f"Features saved to {output_path}")

    return output_df


if __name__ == "__main__":
    run_pipeline()
