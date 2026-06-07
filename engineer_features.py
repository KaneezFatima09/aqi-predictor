"""
feature_pipeline/engineer_features.py
Computes model features and targets from raw data.
Includes time-based features, derived features, and AQI targets.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time-based features to avoid ordinal bias."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"]    = df["timestamp"].dt.hour
    df["dow"]     = df["timestamp"].dt.dayofweek   # 0=Monday
    df["month"]   = df["timestamp"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # Cyclical encoding — sin/cos transforms
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"]   / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"]   / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_lag_features(df: pd.DataFrame, group_col: str = "city") -> pd.DataFrame:
    """Add lagged AQI values and rolling statistics."""
    df = df.sort_values(["city", "timestamp"]).copy()

    for lag in [1, 3, 6, 12, 24]:
        df[f"aqi_lag_{lag}h"] = df.groupby(group_col)["aqi"].shift(lag)

    df["aqi_rolling_mean_6h"]  = df.groupby(group_col)["aqi"].transform(
        lambda x: x.shift(1).rolling(6, min_periods=1).mean()
    )
    df["aqi_rolling_mean_24h"] = df.groupby(group_col)["aqi"].transform(
        lambda x: x.shift(1).rolling(24, min_periods=1).mean()
    )
    df["aqi_rolling_std_6h"]   = df.groupby(group_col)["aqi"].transform(
        lambda x: x.shift(1).rolling(6, min_periods=1).std().fillna(0)
    )
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features: momentum, ratios, interactions."""
    df = df.copy()

    # AQI change rate (momentum): difference from 3-hour lag
    df["aqi_change_rate"] = df["aqi"] - df.get("aqi_lag_3h", df["aqi"])

    # Pollutant ratios
    df["pm_ratio"] = df["pm25"] / (df["pm10"] + 1e-6)

    # Wind × humidity interaction (affects particle suspension)
    df["wind_humidity"] = df["wind_speed"] * df["humidity"] / 100

    # Temperature × humidity (heat index proxy)
    df["heat_index"] = df["temperature"] + 0.33 * (df["humidity"] / 100 * 6.105) - 4

    return df


def compute_targets(df: pd.DataFrame, group_col: str = "city") -> pd.DataFrame:
    """
    Compute prediction targets: AQI at +24h, +48h, +72h.
    These are future AQI values the model learns to predict.
    """
    df = df.sort_values(["city", "timestamp"]).copy()

    for horizon in [24, 48, 72]:
        df[f"target_aqi_{horizon}h"] = df.groupby(group_col)["aqi"].shift(-horizon)

    return df


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

    # Drop rows where targets or key features are NaN (start/end of series)
    df = df.dropna(subset=TARGET_COLS + ["aqi_lag_24h"])
    print(f"  Final rows after dropping NaN: {len(df)}")

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    output_df = df[["city", "timestamp"] + available_features + TARGET_COLS]
    output_df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")
    return output_df


if __name__ == "__main__":
    run_pipeline()
