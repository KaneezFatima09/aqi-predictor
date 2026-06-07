"""
training_pipeline/models.py
Model definitions for AQI forecasting.
Covers statistical baselines → ensemble → deep learning.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── Scikit-learn Models ─────────────────────────────────────────────────────

def get_random_forest(n_estimators=400, max_depth=18, min_samples_split=4,
                      random_state=42):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        n_jobs=-1,
        random_state=random_state,
    )


def get_xgboost():
    if not HAS_XGB:
        raise ImportError("xgboost not installed. Run: pip install xgboost")
    return xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def get_ridge():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0)),
    ])


def get_svr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  MultiOutputRegressor(SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.5))),
    ])


# ── PyTorch LSTM ────────────────────────────────────────────────────────────

if HAS_TORCH:
    class LSTMForecaster(nn.Module):
        """
        LSTM-based AQI forecaster.
        Input:  (batch, seq_len, n_features)
        Output: (batch, n_targets)
        """
        def __init__(self, n_features: int, n_targets: int = 3,
                     hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_targets),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])  # last time-step


    class LSTMWrapper:
        """Sklearn-compatible wrapper around LSTMForecaster."""
        def __init__(self, n_features, n_targets=3, epochs=50, lr=1e-3, batch_size=256):
            self.n_features = n_features
            self.n_targets  = n_targets
            self.epochs     = epochs
            self.lr         = lr
            self.batch_size = batch_size
            self.model      = None
            self.scaler     = StandardScaler()

        def fit(self, X, y):
            X_scaled = self.scaler.fit_transform(X)
            # Reshape for LSTM: (samples, 1, features) — single time-step
            X_t = torch.tensor(X_scaled[:, np.newaxis, :], dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)

            self.model = LSTMForecaster(self.n_features, self.n_targets)
            optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            criterion  = nn.MSELoss()
            dataset    = torch.utils.data.TensorDataset(X_t, y_t)
            loader     = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = criterion(self.model(xb), yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if (epoch + 1) % 10 == 0:
                    print(f"  LSTM Epoch {epoch+1}/{self.epochs} — Loss: {total_loss/len(loader):.4f}")
            return self

        def predict(self, X):
            X_scaled = self.scaler.transform(X)
            X_t = torch.tensor(X_scaled[:, np.newaxis, :], dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                return self.model(X_t).numpy()


def get_all_models(n_features: int):
    models = {
        "Random Forest": get_random_forest(),
        "Ridge Regression": get_ridge(),
        "SVR": get_svr(),
    }
    if HAS_XGB:
        models["XGBoost"] = get_xgboost()
    if HAS_TORCH:
        models["LSTM"] = LSTMWrapper(n_features=n_features)
    return models
