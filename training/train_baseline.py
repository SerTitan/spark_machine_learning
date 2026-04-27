#!/usr/bin/env python3
"""
Обучение baseline моделей для предсказания времени выполнения Spark.

Модели:
- DummyRegressor (median baseline)
- RandomForest + RandomizedSearchCV
- RandomForest + Simulated Annealing
- MLP (128→64)

Использование:
    python training/train_baseline.py --csv ./out/wc_train_all.csv --outdir ./out/baseline

С MLflow:
    python training/train_baseline.py --csv ./out/wc_train_all.csv --outdir ./out/baseline --mlflow
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib


def convert_to_serializable(obj):
    """Конвертирует numpy типы в Python типы для JSON сериализации."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data import create_dataset, SparkDataset
from models.baseline import (
    DummyBaseline,
    RandomForestBaseline,
    SimulatedAnnealingRF,
    MLPBaseline,
    ModelResult,
    train_all_baselines,
    results_to_dataframe,
)


def setup_mlflow(tracking_uri: str, experiment_name: str):
    """Настраивает MLflow."""
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow


def plot_metrics_comparison(results_df: pd.DataFrame, outdir: Path):
    """Строит графики сравнения метрик."""
    metrics = ["MAE", "RMSE", "R2", "MAPE"]

    for metric in metrics:
        if metric not in results_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        results_df[metric].plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Model")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / f"bar_{metric.lower()}.png", dpi=150)
        plt.close()


def plot_sa_convergence(history: list, outdir: Path):
    """Строит график сходимости Simulated Annealing."""
    if not history:
        return
    iters, scores = zip(*history)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, scores, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Simulated Annealing Convergence")
    plt.tight_layout()
    plt.savefig(outdir / "sa_convergence.png", dpi=150)
    plt.close()


def plot_mlp_loss(history: list, outdir: Path):
    """Строит график loss curve MLP."""
    if not history:
        return
    iters, losses = zip(*history)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, losses, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("MLP Training Loss Curve")
    plt.tight_layout()
    plt.savefig(outdir / "mlp_loss_curve.png", dpi=150)
    plt.close()


def plot_predictions_scatter(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, outdir: Path):
    """Строит scatter plot предсказаний vs реальных значений."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual Duration (s)")
    ax.set_ylabel("Predicted Duration (s)")
    ax.set_title(f"{model_name}: Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(outdir / f"scatter_{model_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train baseline models for Spark config optimization")
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--outdir", default="./out/baseline", help="Output directory")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument("--experiment", default="spark_baseline", help="MLflow experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-dummy", action="store_true", help="Skip Dummy baseline")
    parser.add_argument("--rf-search-iters", type=int, default=40, help="RandomizedSearchCV iterations for RF")
    parser.add_argument("--sa-iters", type=int, default=120, help="Simulated Annealing iterations for RF")
    parser.add_argument("--sa-T0", type=float, default=3.0, help="SA initial temperature")
    parser.add_argument("--sa-alpha", type=float, default=0.93, help="SA cooling coefficient")
    parser.add_argument("--mlp-hidden", type=str, default="64,32", help="MLP hidden sizes, comma-separated")
    parser.add_argument("--mlp-lr", type=float, default=0.003, help="MLP learning rate")
    parser.add_argument("--mlp-max-iter", type=int, default=1200, help="MLP max iterations")
    parser.add_argument("--mlp-patience", type=int, default=40, help="MLP early stopping patience (n_iter_no_change)")
    args = parser.parse_args()

    mlp_hidden = tuple(int(x) for x in args.mlp_hidden.split(","))

    # Создаём директории
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Spark Configuration Optimization - Baseline Training")
    print("=" * 60)
    print(f"Dataset: {args.csv}")
    print(f"Output: {outdir}")
    print(f"MLflow: {'enabled' if args.mlflow else 'disabled'}")
    print()

    # Загружаем данные
    print("[1/5] Loading dataset...")
    dataset = create_dataset(args.csv, random_state=args.seed)
    print(f"      Train: {len(dataset.X_train)} samples")
    print(f"      Val:   {len(dataset.X_val)} samples")
    print(f"      Test:  {len(dataset.X_test)} samples")
    print(f"      Features: {dataset.n_features}")
    print()

    # Получаем трансформированные данные
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.get_splits()

    # MLflow setup
    mlflow = None
    if args.mlflow:
        try:
            mlflow = setup_mlflow(args.mlflow_uri, args.experiment)
            print(f"[MLflow] Connected to {args.mlflow_uri}")
            print(f"[MLflow] Experiment: {args.experiment}")
            print()
        except Exception as e:
            print(f"[WARN] MLflow connection failed: {e}")
            print("       Continuing without MLflow...")
            mlflow = None

    # Обучаем модели
    print("[2/5] Training models...")
    results = train_all_baselines(
        X_train, y_train, X_val, y_val, X_test, y_test,
        random_state=args.seed,
        include_dummy=not args.skip_dummy,
        rf_search_iters=args.rf_search_iters,
        sa_iters=args.sa_iters,
        mlp_hidden=mlp_hidden,
        mlp_lr=args.mlp_lr,
        mlp_max_iter=args.mlp_max_iter,
        mlp_patience=args.mlp_patience,
    )
    print()

    # Сохраняем результаты
    print("[3/5] Saving results...")
    results_df = results_to_dataframe(results)
    results_csv = outdir / "metrics_baseline.csv"
    results_df.to_csv(results_csv)
    print(f"      Metrics saved to: {results_csv}")

    # JSON отчёт
    report = {
        "task": Path(args.csv).stem,
        "dataset": str(args.csv),
        "n_train": int(len(dataset.X_train)),
        "n_val": int(len(dataset.X_val)),
        "n_test": int(len(dataset.X_test)),
        "n_features": int(dataset.n_features),
        "models": {}
    }
    for r in results:
        report["models"][r.name] = {
            "metrics": convert_to_serializable(r.metrics),
            "best_params": convert_to_serializable(r.best_params),
        }

    with open(outdir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"      Report saved to: {outdir / 'report.json'}")

    # Сохраняем модели
    for r in results:
        model_path = outdir / f"model_{r.name.lower().replace(' ', '_')}.joblib"
        joblib.dump(r.model, model_path)
    print(f"      Models saved to: {outdir}")

    # Сохраняем препроцессор
    joblib.dump(dataset.preprocessor, outdir / "preprocessor.joblib")
    print(f"      Preprocessor saved to: {outdir / 'preprocessor.joblib'}")

    # Графики
    print("[4/5] Generating plots...")
    plot_metrics_comparison(results_df, plots_dir)

    # SA convergence
    for r in results:
        if r.name == "RandomForest_SimulatedAnnealing" and r.history:
            plot_sa_convergence(r.history, plots_dir)
        if r.name == "MLP" and r.history:
            plot_mlp_loss(r.history, plots_dir)

    # Scatter plots для лучшей модели
    best_result = min(results, key=lambda r: r.metrics["MAE"])
    y_pred_best = best_result.model.predict(X_test)
    plot_predictions_scatter(y_test, y_pred_best, best_result.name, plots_dir)

    # Дополнительно: scatter для RandomForest_RandomSearch если он есть (для сравнения с SA)
    for r in results:
        if r.name == "RandomForest_RandomSearch":
            y_pred_rf_rs = r.model.predict(X_test)
            plot_predictions_scatter(y_test, y_pred_rf_rs, r.name, plots_dir)
            break

    print(f"      Plots saved to: {plots_dir}")

    # MLflow logging
    if mlflow:
        print("[5/5] Logging to MLflow...")
        with mlflow.start_run(run_name="baseline_training"):
            # Parameters
            mlflow.log_params({
                "dataset": str(args.csv),
                "n_train": len(dataset.X_train),
                "n_val": len(dataset.X_val),
                "n_test": len(dataset.X_test),
                "n_features": dataset.n_features,
                "seed": args.seed,
            })

            # Metrics
            for r in results:
                for metric_name, metric_value in r.metrics.items():
                    mlflow.log_metric(f"{r.name}_{metric_name}", metric_value)

            # Artifacts
            mlflow.log_artifact(str(results_csv))
            mlflow.log_artifact(str(outdir / "report.json"))
            for png in plots_dir.glob("*.png"):
                mlflow.log_artifact(str(png))

            # Best model
            mlflow.log_param("best_model", best_result.name)
            mlflow.log_metric("best_MAE", best_result.metrics["MAE"])

        print("      MLflow logging complete")
    else:
        print("[5/5] Skipping MLflow (not configured)")

    # Итоговая таблица
    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(results_df.to_string())
    print()
    print(f"Best model: {best_result.name} (MAE={best_result.metrics['MAE']:.4f})")
    print()
    print(f"[OK] All artifacts saved to: {outdir}")


if __name__ == "__main__":
    main()
