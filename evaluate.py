"""
training_pipeline/evaluate.py
Model evaluation utilities: RMSE/MAE/R², SHAP explainability, and plots.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

Path("models").mkdir(exist_ok=True)


def compute_shap_values(model, X_test: np.ndarray, feature_names: list,
                         max_samples: int = 500) -> None:
    """
    Compute and plot SHAP feature importance values.
    Uses TreeExplainer for tree-based models, KernelExplainer otherwise.
    """
    try:
        import shap
    except ImportError:
        print("  [SHAP] shap not installed — skipping. Run: pip install shap")
        return

    X_sample = X_test[:max_samples]

    try:
        # Fast tree explainer for RF / XGBoost
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        # For multi-output, shap_values is a list — take first target (24h)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    except Exception:
        # Fall back to KernelExplainer (model-agnostic, slower)
        background = shap.kmeans(X_sample, k=20)
        explainer = shap.KernelExplainer(
            lambda x: model.predict(x)[:, 0], background
        )
        shap_values = explainer.shap_values(X_sample[:100])

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#f85149" if mean_abs_shap[i] > 0 else "#3fb950" for i in sorted_idx]
    ax.barh(
        [feature_names[i] for i in sorted_idx][::-1],
        mean_abs_shap[sorted_idx][::-1],
        color=colors[::-1]
    )
    ax.set_xlabel("Mean |SHAP value| (impact on AQI prediction)")
    ax.set_title("SHAP Feature Importance — Top 15 Features (24h Horizon)")
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("models/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  SHAP plot saved to models/shap_importance.png")

    # Print top features
    print("  Top 10 features by SHAP importance:")
    for i in sorted_idx[:10]:
        print(f"    {feature_names[i]:25s}  {mean_abs_shap[i]:.4f}")


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """Scatter plot of predicted vs actual AQI (24h horizon)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    horizons = ["24h", "48h", "72h"]

    for i, (ax, horizon) in enumerate(zip(axes, horizons)):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        ax.scatter(yt, yp, alpha=0.3, s=8, color="#58a6ff")
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, "r--", lw=1, alpha=0.7, label="Perfect")

        from sklearn.metrics import r2_score, mean_squared_error
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2   = r2_score(yt, yp)
        ax.set_title(f"{horizon} Forecast\nRMSE={rmse:.1f}  R²={r2:.3f}", color="white")
        ax.set_xlabel("Actual AQI", color="white")
        ax.set_ylabel("Predicted AQI", color="white")
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(f"Predicted vs Actual AQI — {model_name}", color="white", fontsize=14)
    plt.tight_layout()
    plt.savefig("models/predictions_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Prediction scatter plot saved to models/predictions_scatter.png")
