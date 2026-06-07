"""
feature_pipeline/upload_to_store.py
Uploads engineered features to the Hopsworks Feature Store.
Creates feature groups if they don't exist, upserts otherwise.
"""

import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from engineer_features import FEATURE_COLS, TARGET_COLS

load_dotenv()

HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_KEY     = os.getenv("HOPSWORKS_API_KEY")


def get_feature_store():
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_KEY,
    )
    return project.get_feature_store()


def upload_features(features_path: str = "data/features.csv"):
    print("Connecting to Hopsworks...")
    fs = get_feature_store()

    df = pd.read_csv(features_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["event_time"] = df["timestamp"]  # Hopsworks requires event_time column

    # ── AQI Feature Group ──────────────────────────────────────────────────
    print("Uploading aqi_features_v2...")
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_features_v2",
        version=2,
        primary_key=["city", "timestamp"],
        event_time="event_time",
        description="Engineered AQI and pollutant features, hourly granularity",
        online_enabled=True,
    )
    aqi_fg.insert(df[["city", "timestamp", "event_time"] + FEATURE_COLS])
    print(f"  Inserted {len(df)} rows into aqi_features_v2")

    # ── Target Feature Group ───────────────────────────────────────────────
    print("Uploading aqi_targets_v1...")
    target_fg = fs.get_or_create_feature_group(
        name="aqi_targets_v1",
        version=1,
        primary_key=["city", "timestamp"],
        event_time="event_time",
        description="AQI prediction targets at 24h, 48h, 72h horizons",
    )
    target_fg.insert(df[["city", "timestamp", "event_time"] + TARGET_COLS])
    print(f"  Inserted {len(df)} rows into aqi_targets_v1")

    # ── Feature View ───────────────────────────────────────────────────────
    print("Creating feature view aqi_forecast_fv...")
    try:
        fv = fs.get_feature_view("aqi_forecast_fv", version=1)
        print("  Feature view already exists.")
    except Exception:
        query = aqi_fg.select_all().join(target_fg.select(TARGET_COLS))
        fv = fs.create_feature_view(
            name="aqi_forecast_fv",
            version=1,
            description="Joined features and targets for AQI forecasting",
            query=query,
        )
        print("  Feature view created.")

    print("Upload complete.")
    return fv


if __name__ == "__main__":
    upload_features()
