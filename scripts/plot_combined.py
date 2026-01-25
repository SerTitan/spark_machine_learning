#!/usr/bin/env python3
"""
Генерация сводных графиков по метрикам и scatter для RF и DNN.

Пример:
    python scripts/plot_combined.py \
        --baseline-dir out/final_best/baseline \
        --dnn-dir out/final_best/dnn \
        --csv data/wc_train_merged.csv \
        --seed 42 \
        --outdir out/final_best/plots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.data import create_dataset  # noqa: E402
from models.dnn_predictor import DNNPredictor  # noqa: E402


def plot_panels(metrics_df: pd.DataFrame, outdir: Path):
    labels = {
        "MAE": "Mean absolute error (s)",
        "RMSE": "Root mean square error (s)",
        "R2": "R-Square",
        "MAPE": "Mean absolute percent error (%)",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric in zip(axes.flat, ["MAE", "RMSE", "R2", "MAPE"]):
        metrics_df[metric].sort_values().plot(
            kind="bar", ax=ax, color="skyblue", edgecolor="black"
        )
        ax.set_ylabel(labels[metric])
        ax.set_xlabel("Model")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title(labels[metric])
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outdir / "panel_metrics.png", dpi=180)
    plt.close()


def plot_scatter(y_true, preds: dict[str, pd.Series], outdir: Path):
    lims = [
        min([y_true.min()] + [v.min() for v in preds.values()]),
        max([y_true.max()] + [v.max() for v in preds.values()]),
    ]
    fig, ax = plt.subplots(figsize=(7, 7))
    for name, vals in preds.items():
        ax.scatter(y_true, vals, s=30, alpha=0.6, label=name)
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual duration (s)")
    ax.set_ylabel("Predicted duration (s)")
    ax.set_title("Predicted vs Actual")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "scatter_all.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot combined baseline/DNN metrics and scatters.")
    parser.add_argument("--baseline-dir", required=True, help="Directory with baseline outputs (metrics_baseline.csv).")
    parser.add_argument("--dnn-dir", help="Directory with DNN outputs (report.json and model).")
    parser.add_argument("--csv", required=True, help="Dataset CSV for scatter plotting.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset split.")
    parser.add_argument("--outdir", help="Output directory for plots (default: <baseline-dir>/plots_combined)")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    outdir = Path(args.outdir) if args.outdir else baseline_dir / "plots_combined"
    outdir.mkdir(parents=True, exist_ok=True)

    # Load metrics
    metrics = pd.read_csv(baseline_dir / "metrics_baseline.csv", index_col="model")
    if args.dnn_dir:
        dnn_dir = Path(args.dnn_dir)
        with open(dnn_dir / "report.json") as f:
            dnn_metrics = json.load(f)["metrics"]["test"]
        metrics.loc["DNN"] = [
            dnn_metrics["MAE"],
            dnn_metrics["RMSE"],
            dnn_metrics["R2"],
            dnn_metrics["MAPE"],
        ]

    plot_panels(metrics, outdir)

    # Scatter: require models and data
    dataset = create_dataset(args.csv, random_state=args.seed)
    _, _, X_test, _, _, y_test = dataset.get_splits()
    preds = {}

    # RF SA
    rf_sa_path = baseline_dir / "model_randomforest_simulatedannealing.joblib"
    if rf_sa_path.exists():
        rf_sa = joblib.load(rf_sa_path)
        preds["RF_SA"] = rf_sa.predict(X_test)

    # RF RS
    rf_rs_path = baseline_dir / "model_randomforest_randomsearch.joblib"
    if rf_rs_path.exists():
        rf_rs = joblib.load(rf_rs_path)
        preds["RF_RS"] = rf_rs.predict(X_test)

    # DNN
    if args.dnn_dir:
        dnn = DNNPredictor.load(Path(args.dnn_dir) / "model")
        preds["DNN"] = dnn.predict(X_test)

    if preds:
        plot_scatter(y_test, preds, outdir)

    print(f"[OK] Plots saved to: {outdir}")


if __name__ == "__main__":
    main()
