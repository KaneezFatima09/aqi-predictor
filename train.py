"""
training_pipeline/train.py
Trains all ML models, evaluates on holdout test set,
selects best model, and registers it in the Hopsworks Model Registry.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv

from models import get_all_models
from evaluate import compute_shap_values, plot_predictions

load_dotenv()

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


def load_data_from_csv(path: str = "data/features.csv"):
    df = pd.read_csv(path).dropna(subset=FEATURE_COLS + TARGET_COLS)
    X = df[FEATURE_COLS].values.astype(float)
    y = df[TARGET_COLS].values.astype(float)
    return X, y, df


def load_data_from_hopsworks():
    import hopsworks
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )
    fs = project.get_feature_store()
    fv = fs.get_feature_view("aqi_forecast_fv", version=1)
    df, _ = fv.training_data(description="AQI forecast training data")
    df = df.dropna(subset=FEATURE_COLS + TARGET_COLS)
    X = df[FEATURE_COLS].values.astype(float)
    y = df[TARGET_COLS].values.astype(float)
    return X, y, df


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    # Average metrics across all three forecast horizons
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))
    print(f"  {name:20s} | RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}")
    return {"name": name, "rmse": rmse, "mae": mae, "r2": r2, "model": model, "y_pred": y_pred}


def register_model(model, metrics: dict, feature_cols: list):
    """Save model artifact and register in Hopsworks Model Registry."""
    import hopsworks
    Path("models").mkdir(exist_ok=True)

    # Save locally
    joblib.dump(model, "models/best_model.pkl")
    with open("models/metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if k not in ("model", "y_pred")}, f, indent=2)
    with open("models/features.json", "w") as f:
        json.dump(feature_cols, f)

    # Register in Hopsworks
    try:
        project = hopsworks.login(
            project=os.getenv("HOPSWORKS_PROJECT"),
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        )
        mr = project.get_model_registry()
        model_obj = mr.sklearn.create_model(
            name="aqi_random_forest",
            metrics={"rmse": metrics["rmse"], "mae": metrics["mae"], "r2": metrics["r2"]},
            description=f"AQI 3-day forecaster — trained {datetime.utcnow().date()}",
            input_example=np.zeros((1, len(feature_cols))),
            feature_view=None,
        )
        model_obj.save("models/")
        print(f"Model registered in Hopsworks Model Registry (RMSE={metrics['rmse']:.2f})")
    except Exception as e:
        print(f"[Warning] Could not register in Hopsworks: {e}")
        print("Model saved locally at models/best_model.pkl")


def run_training(use_hopsworks: bool = False):
    print("=" * 60)
    print("Pearls AQI Predictor — Training Pipeline")
    print("=" * 60)

    # Load data
    if use_hopsworks:
        print("Loading data from Hopsworks Feature Store...")
        X, y, df = load_data_from_hopsworks()
    else:
        print("Loading data from CSV...")
        X, y, df = load_data_from_csv()

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} targets")

    # Time-based split — last 30 days as test set
    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    # Train and evaluate all models
    print("\nTraining models...")
    models = get_all_models(n_features=X.shape[1])
    results = []

    for name, model in models.items():
        print(f"\n[{name}]")
        try:
            model.fit(X_train, y_train)
            result = evaluate_model(model, X_test, y_test, name)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Select best model by RMSE
    best = min(results, key=lambda r: r["rmse"])
    print(f"\n{'='*60}")
    print(f"Best model: {best['name']}  (RMSE={best['rmse']:.2f}, R²={best['r2']:.3f})")

    # SHAP analysis on best model
    print("\nComputing SHAP values...")
    compute_shap_values(best["model"], X_test, FEATURE_COLS)

    # Plots
    print("Generating evaluation plots...")
    plot_predictions(y_test, best["y_pred"], best["name"])

    # Register
    print("\nRegistering model...")
    register_model(best["model"], best, FEATURE_COLS)

    print("\nTraining pipeline complete.")
    return best


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hopsworks", action="store_true", help="Load data from Hopsworks")
    args = parser.parse_args()
    run_training(use_hopsworks=args.hopsworks)
