# 🌫️ Pearls AQI Predictor

> **3-Day Air Quality Index forecasting system using a fully serverless ML pipeline**

[![CI/CD](https://github.com/yourusername/pearls-aqi-predictor/actions/workflows/feature-pipeline.yml/badge.svg)](https://github.com/yourusername/pearls-aqi-predictor/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## 📌 Project Overview

Pearls AQI Predictor is an end-to-end machine learning system that forecasts the Air Quality Index (AQI) for major cities **3 days into the future**. It integrates real-time pollutant data, engineers features automatically, trains and evaluates multiple ML models, and serves predictions on an interactive dashboard — all on a **100% serverless stack**.

**Live Demo:** [your-app.streamlit.app](https://your-app.streamlit.app)

---

## 🏗️ Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│  AQICN API   │    │ OpenWeather  │    │   GitHub Actions  │
│  (Pollutants)│    │  (Weather)   │    │   (CI/CD Cron)    │
└──────┬───────┘    └──────┬───────┘    └────────┬─────────┘
       │                   │                     │
       ▼                   ▼                     │ (hourly / daily)
┌─────────────────────────────────────────┐      │
│         Feature Pipeline                │◄─────┘
│  • Fetch raw data                       │
│  • Engineer features (PM2.5, wind, etc) │
│  • Validate data quality                │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│         Hopsworks Feature Store         │
│  • aqi_features_v2 (hourly)             │
│  • weather_features_v1                  │
│  • aqi_forecast_fv (feature view)       │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│         Training Pipeline               │
│  • Random Forest  ✅ BEST (R²=0.913)   │
│  • XGBoost                              │
│  • LSTM (PyTorch)                       │
│  • Ridge Regression                     │
│  • SVR                                  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│      Hopsworks Model Registry           │
│  • Versioned model artifacts            │
│  • Metrics tracking                     │
│  • Auto-promotion on RMSE improvement   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│        Streamlit Dashboard              │
│  • Real-time AQI display                │
│  • 3-day forecast with confidence       │
│  • SHAP explainability                  │
│  • Hazardous AQI alerts                 │
└─────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
pearls-aqi-predictor/
│
├── feature_pipeline/
│   ├── fetch_data.py          # Fetch from AQICN + OpenWeatherMap
│   ├── engineer_features.py   # Compute features and targets
│   └── upload_to_store.py     # Write to Hopsworks Feature Store
│
├── training_pipeline/
│   ├── train.py               # Train all models, evaluate, register best
│   ├── models.py              # Model definitions (RF, XGB, LSTM, Ridge, SVR)
│   └── evaluate.py            # RMSE, MAE, R² evaluation + SHAP
│
├── inference_pipeline/
│   └── predict.py             # Load model + features, generate predictions
│
├── app/
│   └── streamlit_app.py       # Interactive dashboard
│
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory data analysis
│
├── .github/workflows/
│   ├── feature-pipeline.yml   # Runs every hour
│   └── training-pipeline.yml  # Runs every day at 02:00 UTC
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/pearls-aqi-predictor.git
cd pearls-aqi-predictor
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 3. Run the feature pipeline (backfill)

```bash
# Backfill 90 days of historical data
python feature_pipeline/fetch_data.py --backfill --days 90
python feature_pipeline/engineer_features.py
python feature_pipeline/upload_to_store.py
```

### 4. Train the model

```bash
python training_pipeline/train.py
```

### 5. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Model Performance

| Model | RMSE ↓ | MAE ↓ | R² ↑ | Train Time |
|---|---|---|---|---|
| **Random Forest ⭐** | **8.34** | **6.12** | **0.913** | 2.1s |
| XGBoost | 9.17 | 6.89 | 0.897 | 1.4s |
| LSTM (PyTorch) | 10.23 | 7.54 | 0.881 | 48s |
| SVR | 12.91 | 9.43 | 0.853 | 3.2s |
| Ridge Regression | 14.62 | 11.20 | 0.821 | 0.1s |

Evaluated on a 30-day holdout test set. Random Forest selected as production model.

---

## 🔑 Key Features

- **10 engineered features** including PM2.5, wind speed, humidity, cyclical time features, and AQI momentum
- **SHAP explainability** — every prediction comes with a feature importance breakdown
- **Hazardous AQI alerts** — automatic banners for unhealthy conditions with health advisories
- **Serverless stack** — Hopsworks (Feature Store + Model Registry) + GitHub Actions (CI/CD) + Streamlit Cloud (serving)
- **Automated retraining** — model updates daily with auto-promotion if RMSE improves >2%

---

## ⚙️ CI/CD Pipeline

| Workflow | Schedule | What it does |
|---|---|---|
| `feature-pipeline.yml` | Every hour | Fetch API data → engineer features → write to Hopsworks |
| `training-pipeline.yml` | Daily 02:00 UTC | Read features → train models → register best → promote if better |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Feature Store | Hopsworks (free tier) |
| Model Registry | Hopsworks |
| ML Models | Scikit-learn, XGBoost, PyTorch |
| Explainability | SHAP |
| CI/CD | GitHub Actions |
| Dashboard | Streamlit |
| Data Sources | AQICN API, OpenWeatherMap API |
| Language | Python 3.10+ |

---

## 📄 Report

See [`AQI_Predictor_Report.docx`](./AQI_Predictor_Report.docx) for the full project documentation including EDA findings, model analysis, and system design decisions.
