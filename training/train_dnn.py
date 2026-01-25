#!/usr/bin/env python3
"""
Обучение DNN Performance Predictor для предсказания времени выполнения Spark.

Архитектура по статье:
    Input(n_features) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(1)

Использование:
    python training/train_dnn.py --csv ./out/wc_train_all.csv --outdir ./out/dnn

С MLflow:
    python training/train_dnn.py --csv ./out/wc_train_all.csv --outdir ./out/dnn --mlflow
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data import create_dataset
from models.dnn_predictor import DNNPredictor, DNNConfig, create_dnn_predictor
from models.baseline import compute_metrics


def setup_mlflow(tracking_uri: str, experiment_name: str):
    """Настраивает MLflow."""
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow


def plot_learning_curves(history, outdir: Path):
    """Строит графики обучения."""
    # Loss curve
    if history.train_loss:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.train_loss, label="Train Loss", alpha=0.8)
        if history.val_loss:
            ax.plot(history.val_loss, label="Val Loss", alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title("Training Loss Curve")
        ax.legend()
        ax.axvline(x=history.best_epoch, color="red", linestyle="--", label="Best epoch")
        plt.tight_layout()
        plt.savefig(outdir / "loss_curve.png", dpi=150)
        plt.close()

    # RMSE curve
    if history.val_rmse:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.val_rmse, label="Val RMSE", color="orange")
        ax.axhline(y=history.best_val_rmse, color="red", linestyle="--", label=f"Best: {history.best_val_rmse:.2f}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE (seconds)")
        ax.set_title("Validation RMSE Curve")
        ax.legend()
        plt.tight_layout()
        plt.savefig(outdir / "val_rmse_curve.png", dpi=150)
        plt.close()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, outdir: Path):
    """Строит scatter plot предсказаний."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual Duration (s)")
    ax.set_ylabel("Predicted Duration (s)")
    ax.set_title("DNN: Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(outdir / "predictions_scatter.png", dpi=150)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, outdir: Path):
    """Строит гистограмму остатков."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--")
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residuals Distribution (std={np.std(residuals):.2f})")
    plt.tight_layout()
    plt.savefig(outdir / "residuals_hist.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train DNN Performance Predictor")
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--outdir", default="./out/dnn", help="Output directory")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument("--experiment", default="spark_dnn", help="MLflow experiment name")

    # Model hyperparameters
    parser.add_argument("--hidden-sizes", type=str, default="128,64",
                        help="Hidden layer sizes, comma-separated (default: 128,64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=500, help="Max epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-log-target", action="store_true",
                        help="Don't use log1p transform on target")

    args = parser.parse_args()

    # Parse hidden sizes
    hidden_sizes = tuple(int(x) for x in args.hidden_sizes.split(","))

    # Создаём директории
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Spark Configuration Optimization - DNN Training")
    print("=" * 60)
    print(f"Dataset: {args.csv}")
    print(f"Output: {outdir}")
    print(f"Architecture: Input → {' → '.join(map(str, hidden_sizes))} → 1")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.epochs}")
    print(f"Patience: {args.patience}")
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
            mlflow = None

    # Создаём и обучаем модель
    print("[2/5] Training DNN...")
    config = DNNConfig(
        hidden_sizes=hidden_sizes,
        dropout=args.dropout,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        use_log_target=not args.no_log_target,
        random_state=args.seed,
    )
    predictor = DNNPredictor(config)
    predictor.fit(X_train, y_train, X_val, y_val, verbose=True)
    print()

    # Evaluate
    print("[3/5] Evaluating...")
    y_pred_test = predictor.predict(X_test)
    y_pred_val = predictor.predict(X_val)

    test_metrics = compute_metrics(y_test, y_pred_test)
    val_metrics = compute_metrics(y_val, y_pred_val)

    print(f"      Validation: MAE={val_metrics['MAE']:.4f}, RMSE={val_metrics['RMSE']:.4f}, R²={val_metrics['R2']:.4f}")
    print(f"      Test:       MAE={test_metrics['MAE']:.4f}, RMSE={test_metrics['RMSE']:.4f}, R²={test_metrics['R2']:.4f}")
    print()

    # Сохраняем модель
    print("[4/5] Saving model and artifacts...")
    predictor.save(outdir / "model")
    print(f"      Model saved to: {outdir / 'model'}")

    # Report
    report = {
        "dataset": str(args.csv),
        "n_train": len(dataset.X_train),
        "n_val": len(dataset.X_val),
        "n_test": len(dataset.X_test),
        "n_features": dataset.n_features,
        "config": {
            "hidden_sizes": hidden_sizes,
            "dropout": args.dropout,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "use_log_target": not args.no_log_target,
        },
        "training": {
            "best_epoch": predictor.history.best_epoch,
            "best_val_rmse": predictor.history.best_val_rmse,
            "total_epochs": len(predictor.history.train_loss),
        },
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
        },
    }
    with open(outdir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"      Report saved to: {outdir / 'report.json'}")

    # Графики
    plot_learning_curves(predictor.history, plots_dir)
    plot_predictions(y_test, y_pred_test, plots_dir)
    plot_residuals(y_test, y_pred_test, plots_dir)
    print(f"      Plots saved to: {plots_dir}")

    # MLflow
    if mlflow:
        print("[5/5] Logging to MLflow...")
        with mlflow.start_run(run_name="dnn_training"):
            # Parameters
            mlflow.log_params({
                "hidden_sizes": str(hidden_sizes),
                "dropout": args.dropout,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "max_epochs": args.epochs,
                "patience": args.patience,
                "use_log_target": not args.no_log_target,
                "n_train": len(dataset.X_train),
                "n_val": len(dataset.X_val),
                "n_test": len(dataset.X_test),
            })

            # Metrics
            for name, value in test_metrics.items():
                mlflow.log_metric(f"test_{name}", value)
            for name, value in val_metrics.items():
                mlflow.log_metric(f"val_{name}", value)
            mlflow.log_metric("best_epoch", predictor.history.best_epoch)
            mlflow.log_metric("best_val_rmse", predictor.history.best_val_rmse)

            # Artifacts
            mlflow.log_artifact(str(outdir / "report.json"))
            for png in plots_dir.glob("*.png"):
                mlflow.log_artifact(str(png))

        print("      MLflow logging complete")
    else:
        print("[5/5] Skipping MLflow (not configured)")

    # Summary
    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Best epoch: {predictor.history.best_epoch}")
    print(f"Best val RMSE: {predictor.history.best_val_rmse:.4f}")
    print()
    print("Test Metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
    print()
    print(f"[OK] All artifacts saved to: {outdir}")


if __name__ == "__main__":
    main()
