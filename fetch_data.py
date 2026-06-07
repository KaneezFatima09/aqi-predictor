"""
feature_pipeline/fetch_data.py
Fetches raw AQI and weather data from external APIs.
Supports both real-time and historical backfill modes.
"""

import os
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")

CITIES = {
    "lahore":  {"lat": 31.5204, "lon": 74.3587, "station": "@7079"},
    "delhi":   {"lat": 28.6139, "lon": 77.2090, "station": "@7026"},
    "beijing": {"lat": 39.9042, "lon": 116.4074,"station": "@1451"},
    "dubai":   {"lat": 25.2048, "lon": 55.2708, "station": "@9007"},
    "london":  {"lat": 51.5074, "lon": -0.1278, "station": "@5765"},
    "new_york":{"lat": 40.7128, "lon": -74.0060,"station": "@5672"},
}


def fetch_aqi(city: str) -> dict:
    """Fetch current AQI data from AQICN API."""
    station = CITIES[city]["station"]
    url = f"https://api.waqi.info/feed/{station}/?token={AQICN_TOKEN}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()["data"]

    iaqi = data.get("iaqi", {})
    return {
        "city": city,
        "timestamp": datetime.utcnow().isoformat(),
        "aqi": data.get("aqi"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "o3":   iaqi.get("o3",   {}).get("v"),
        "no2":  iaqi.get("no2",  {}).get("v"),
        "co":   iaqi.get("co",   {}).get("v"),
        "so2":  iaqi.get("so2",  {}).get("v"),
    }


def fetch_weather(city: str) -> dict:
    """Fetch current weather from OpenWeatherMap API."""
    lat = CITIES[city]["lat"]
    lon = CITIES[city]["lon"]
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    return {
        "city": city,
        "timestamp": datetime.utcnow().isoformat(),
        "temperature": data["main"]["temp"],
        "humidity":    data["main"]["humidity"],
        "pressure":    data["main"]["pressure"],
        "wind_speed":  data["wind"]["speed"] * 3.6,  # m/s → km/h
        "wind_deg":    data["wind"].get("deg", 0),
        "visibility":  data.get("visibility", 10000) / 1000,  # m → km
        "description": data["weather"][0]["description"],
    }


def fetch_all_cities() -> pd.DataFrame:
    """Fetch AQI + weather for all configured cities."""
    records = []
    for city in CITIES:
        try:
            aqi_data = fetch_aqi(city)
            weather_data = fetch_weather(city)
            merged = {**aqi_data, **{k: v for k, v in weather_data.items() if k not in ("city", "timestamp")}}
            records.append(merged)
            print(f"[OK] {city}: AQI={merged['aqi']}, PM2.5={merged['pm25']}, Temp={merged['temperature']}°C")
        except Exception as e:
            print(f"[ERROR] {city}: {e}")
    return pd.DataFrame(records)


def backfill(days: int = 90) -> pd.DataFrame:
    """
    Simulate historical backfill for training data.
    In production this would call a historical API endpoint.
    Here we generate realistic synthetic data for demonstration.
    """
    import numpy as np
    np.random.seed(42)

    records = []
    base_time = datetime.utcnow() - timedelta(days=days)

    for city, info in CITIES.items():
        # City-specific baseline AQI
        baselines = {
            "lahore": 160, "delhi": 200, "beijing": 130,
            "dubai": 85, "london": 40, "new_york": 55
        }
        base_aqi = baselines[city]

        for hour_offset in range(days * 24):
            ts = base_time + timedelta(hours=hour_offset)
            hour = ts.hour
            month = ts.month

            # Diurnal pattern: higher at rush hours
            diurnal = 30 * np.sin((hour - 3) * np.pi / 12) if 6 <= hour <= 22 else -10
            # Seasonal: worse in winter
            seasonal = 20 * np.cos((month - 7) * np.pi / 6)
            noise = np.random.normal(0, 12)

            aqi = max(10, int(base_aqi + diurnal + seasonal + noise))
            pm25 = round(aqi * 0.37 + np.random.normal(0, 3), 1)

            records.append({
                "city": city,
                "timestamp": ts.isoformat(),
                "aqi": aqi,
                "pm25": pm25,
                "pm10": round(pm25 * 1.6, 1),
                "o3": round(aqi * 0.21 + np.random.normal(0, 2), 1),
                "no2": round(aqi * 0.18 + np.random.normal(0, 2), 1),
                "co": round(aqi * 0.05 + np.random.normal(0, 0.5), 2),
                "so2": round(aqi * 0.03 + np.random.normal(0, 1), 1),
                "temperature": round(25 + seasonal * 0.5 + np.random.normal(0, 4), 1),
                "humidity": round(max(10, min(95, 55 + np.random.normal(0, 15))), 1),
                "pressure": round(1013 + np.random.normal(0, 8), 1),
                "wind_speed": round(max(0, 12 + np.random.normal(0, 5)), 1),
                "wind_deg": int(np.random.randint(0, 360)),
                "visibility": round(max(1, 10 + np.random.normal(0, 3)), 1),
            })

    df = pd.DataFrame(records)
    df.to_csv("data/raw_backfill.csv", index=False)
    print(f"[Backfill] Generated {len(df)} records for {days} days across {len(CITIES)} cities.")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true", help="Run historical backfill")
    parser.add_argument("--days", type=int, default=90, help="Days to backfill")
    args = parser.parse_args()

    if args.backfill:
        df = backfill(days=args.days)
    else:
        df = fetch_all_cities()
        df.to_csv("data/latest_raw.csv", index=False)
        print(df)
