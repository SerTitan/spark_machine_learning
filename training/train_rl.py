#!/usr/bin/env python3
"""
Обучение RL-оптимизаторов для поиска оптимальных параметров Spark.

Алгоритмы:
- Q-Learning (табличный)
- DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)
- Bayesian Optimization

Использование:
    # Сначала обучить DNN predictor
    python training/train_dnn.py --csv ./out/wc_train_all.csv --outdir ./out/dnn

    # Затем запустить RL оптимизацию
    python training/train_rl.py \\
        --csv ./out/wc_train_all.csv \\
        --dnn-model ./out/dnn/model \\
        --outdir ./out/rl

    # С конкретным алгоритмом
    python training/train_rl.py --algorithm qlearning ...
    python training/train_rl.py --algorithm dqn ...
    python training/train_rl.py --algorithm ppo ...
    python training/train_rl.py --algorithm bayesian ...
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
import pandas as pd
import joblib

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data import create_dataset, get_default_config
from models.dnn_predictor import DNNPredictor
from models.rl_optimizer import (
    TabularQLearning,
    DQNOptimizer,
    BayesianOptimizer,
    StableBaselinesOptimizer,
    OptimizationResult,
    run_all_optimizers,
)


def setup_mlflow(tracking_uri: str, experiment_name: str):
    """Настраивает MLflow."""
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow


def plot_optimization_trajectory(result: OptimizationResult, outdir: Path):
    """Строит график траектории оптимизации."""
    if result.history.empty or "predicted_time" not in result.history.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    times = result.history["predicted_time"].values
    best_so_far = np.minimum.accumulate(times)

    ax.plot(times, alpha=0.3, label="Current", color="blue")
    ax.plot(best_so_far, linewidth=2, label="Best so far", color="red")
    ax.axhline(y=result.best_predicted_time, linestyle="--", color="green", label=f"Best: {result.best_predicted_time:.2f}s")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Predicted Time (s)")
    ax.set_title(f"{result.algorithm}: Optimization Trajectory")
    ax.legend()

    plt.tight_layout()
    plt.savefig(outdir / f"trajectory_{result.algorithm.lower()}.png", dpi=150)
    plt.close()


def plot_comparison(results: dict, baseline_time: float, outdir: Path):
    """Строит сравнительный график алгоритмов."""
    if not results:
        return

    algorithms = list(results.keys())
    times = [results[a].best_predicted_time for a in algorithms]
    speedups = [baseline_time / t if t > 0 else 0 for t in times]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Times
    bars1 = ax1.bar(algorithms, times, color="steelblue", edgecolor="black")
    ax1.axhline(y=baseline_time, linestyle="--", color="red", label=f"Baseline: {baseline_time:.2f}s")
    ax1.set_ylabel("Predicted Time (s)")
    ax1.set_title("Best Predicted Time by Algorithm")
    ax1.legend()
    for bar, t in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{t:.1f}s",
                ha="center", va="bottom", fontsize=9)

    # Speedups
    bars2 = ax2.bar(algorithms, speedups, color="forestgreen", edgecolor="black")
    ax2.axhline(y=1.0, linestyle="--", color="red", label="Baseline (1.0x)")
    ax2.set_ylabel("Speedup vs Baseline")
    ax2.set_title("Speedup by Algorithm")
    ax2.legend()
    for bar, s in zip(bars2, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{s:.2f}x",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(outdir / "comparison.png", dpi=150)
    plt.close()


def create_predictor_wrapper(dnn_predictor: DNNPredictor, dataset):
    """Создаёт wrapper для предиктора."""
    def predict(df: pd.DataFrame) -> np.ndarray:
        # Ensure all required columns exist
        for col in dataset.X_train.columns:
            if col not in df.columns:
                # Use default value
                if col in dataset.numeric_cols:
                    df[col] = dataset.X_train[col].median()
                elif col in dataset.boolean_cols:
                    df[col] = int(dataset.X_train[col].mode().iloc[0])
                else:
                    df[col] = dataset.X_train[col].mode().iloc[0]

        X = df[dataset.X_train.columns]
        X_transformed = dataset.preprocessor.transform(X)
        return dnn_predictor.predict(X_transformed)

    return predict


def snap_topology(topology: dict, df_topos: pd.DataFrame) -> dict:
    """Приводит топологию к ближайшей из представленных в датасете (по L1)."""
    if df_topos.empty:
        return topology
    cols = ["topology_workers", "topology_worker_cores", "topology_worker_mem_gb"]
    target = np.array([topology[c] for c in cols], dtype=float)
    arr = df_topos[cols].values.astype(float)
    dists = np.abs(arr - target).sum(axis=1)
    best = df_topos.iloc[int(np.argmin(dists))]
    snapped = topology.copy()
    for c in cols:
        snapped[c] = best[c]
    return snapped


def clean_number(v):
    """Убирает .0 если число целое."""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def memory_to_mb(val: str) -> float:
    s = str(val).strip().lower()
    if s.endswith("g"):
        return float(s[:-1]) * 1024.0
    if s.endswith("m"):
        return float(s[:-1])
    if s.endswith("k"):
        return float(s[:-1]) / 1024.0
    return float(s)


def memory_to_kb(val: str) -> float:
    mb = memory_to_mb(val)
    return mb * 1024.0


def pick_baseline_from_topology(dataset, topology: dict):
    """
    Берёт конфиг не лучший: по медиане выбранной топологии (ближайшая к медиане строка).
    Возвращает (config_dict, median_time_from_data).
    """
    df = dataset.raw_df.copy()
    topo_mask = (
        (df["topology_workers"] == topology["topology_workers"]) &
        (df["topology_worker_cores"] == topology["topology_worker_cores"]) &
        (df["topology_worker_mem_gb"] == topology["topology_worker_mem_gb"])
    )
    df = df[topo_mask].copy()
    if df.empty:
        return None, None
    df = df[df["exit_code"] == 0]
    df = df[pd.to_numeric(df["median_duration_s"], errors="coerce") > 0]
    med = df["median_duration_s"].median()
    # ближайшая к медиане строка
    row = df.iloc[(df["median_duration_s"] - med).abs().argsort().iloc[0]].copy()

    # подготовим к числам
    cfg = {
        "executor_cores": int(row["executor_cores"]),
        "executor_instances": int(row["executor_instances"]),
        "driver_cores": int(row["driver_cores"]),
        "executor_memory_mb": memory_to_mb(row["executor_memory"]),
        "driver_memory_mb": memory_to_mb(row["driver_memory"]),
        "memory_fraction": float(row["memory_fraction"]),
        "memory_storageFraction": float(row["memory_storageFraction"]),
        "rpc_message_maxSize": int(row["rpc_message_maxSize"]),
        "shuffle_file_buffer_kb": memory_to_kb(row["shuffle_file_buffer"]),
        "broadcast_block_mb": memory_to_mb(row["broadcast_block"]),
        "maxSizeInFlight_mb": memory_to_mb(row["maxSizeInFlight"]),
        "shuffle_compress": int(bool(row["shuffle_compress"])) if row["shuffle_compress"] in [0,1,True,False] else int(str(row["shuffle_compress"]).lower() in ("true","1")),
        "spill_compress": int(bool(row["spill_compress"])) if row["spill_compress"] in [0,1,True,False] else int(str(row["spill_compress"]).lower() in ("true","1")),
        "broadcast_compress": int(bool(row["broadcast_compress"])) if row["broadcast_compress"] in [0,1,True,False] else int(str(row["broadcast_compress"]).lower() in ("true","1")),
        "rdd_compress": int(bool(row["rdd_compress"])) if row["rdd_compress"] in [0,1,True,False] else int(str(row["rdd_compress"]).lower() in ("true","1")),
        "io_codec": row["io_codec"],
    }
    return cfg, float(med)


def main():
    parser = argparse.ArgumentParser(description="Train RL optimizers for Spark config")
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument("--predictor-type", choices=["dnn", "rf"], default="dnn",
                        help="Which predictor to use: dnn (DNNPredictor) or rf (RandomForest_SA from baseline)")
    parser.add_argument("--model-dir", required=True, help="Path to trained model directory (for dnn: DNNPredictor dir; for rf: baseline out dir containing model_randomforest_simulatedannealing.joblib and preprocessor.joblib)")
    parser.add_argument("--outdir", default="./out/rl", help="Output directory")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument("--experiment", default="spark_rl", help="MLflow experiment name")

    # Algorithm selection
    parser.add_argument("--algorithm", choices=["all", "qlearning", "dqn", "ppo", "bayesian"],
                        default="all", help="Algorithm to run")

    # Q-Learning params
    parser.add_argument("--ql-episodes", type=int, default=100, help="Q-Learning episodes")
    parser.add_argument("--ql-alpha", type=float, default=0.1, help="Q-Learning learning rate")
    parser.add_argument("--ql-gamma", type=float, default=0.95, help="Q-Learning discount factor")
    parser.add_argument("--ql-epsilon", type=float, default=0.3, help="Q-Learning initial epsilon")

    # DQN params
    parser.add_argument("--dqn-episodes", type=int, default=500, help="DQN episodes")

    # PPO params
    parser.add_argument("--ppo-timesteps", type=int, default=10000, help="PPO total timesteps")

    # Bayesian params
    parser.add_argument("--bo-trials", type=int, default=100, help="Bayesian optimization trials")

    # Topology (for optimization target)
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--worker-cores", type=int, default=2, help="Cores per worker")
    parser.add_argument("--worker-mem", type=int, default=4, help="Memory per worker (GB)")
    parser.add_argument("--profile", default="large", help="Workload profile")
    parser.add_argument("--baseline-from-topology", action="store_true",
                        help="Use median run from chosen topology as baseline/default config")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Создаём директории
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Spark Configuration Optimization - RL Training")
    print("=" * 60)
    print(f"Dataset: {args.csv}")
    print(f"Predictor: {args.predictor_type}")
    print(f"Model dir: {args.model_dir}")
    print(f"Output: {outdir}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Topology (requested): {args.workers} workers × {args.worker_cores} cores × {args.worker_mem}GB")
    print(f"Profile: {args.profile}")
    print()

    # Загружаем данные
    print("[1/5] Loading dataset and predictor...")
    dataset = create_dataset(args.csv, random_state=args.seed)
    print(f"      Dataset: {len(dataset.X_train)} train samples")

    if args.predictor_type == "dnn":
        predictor = DNNPredictor.load(args.model_dir)
        print(f"      DNN model loaded from: {args.model_dir}")
        # Создаём predictor wrapper
        predict_fn = create_predictor_wrapper(predictor, dataset)
    else:
        # RF + preprocessor
        rf_path = Path(args.model_dir) / "model_randomforest_simulatedannealing.joblib"
        preproc_path = Path(args.model_dir) / "preprocessor.joblib"
        if not rf_path.exists() or not preproc_path.exists():
            raise FileNotFoundError("RF predictor requires model_randomforest_simulatedannealing.joblib and preprocessor.joblib")
        rf_model = joblib.load(rf_path)
        preproc = joblib.load(preproc_path)

        def predict_fn(df: pd.DataFrame) -> np.ndarray:
            # align columns
            for col in dataset.X_train.columns:
                if col not in df.columns:
                    if col in dataset.numeric_cols:
                        df[col] = dataset.X_train[col].median()
                    elif col in dataset.boolean_cols:
                        df[col] = int(dataset.X_train[col].mode().iloc[0])
                    else:
                        df[col] = dataset.X_train[col].mode().iloc[0]
            X = df[dataset.X_train.columns]
            Xt = preproc.transform(X)
            return rf_model.predict(Xt)

        print(f"      RF model loaded from: {rf_path}")
    print()

    # Topology (snap to nearest seen in data)
    topology_requested = {
        "topology_workers": args.workers,
        "topology_worker_cores": args.worker_cores,
        "topology_worker_mem_gb": args.worker_mem,
    }
    topo_unique = dataset.raw_df[["topology_workers","topology_worker_cores","topology_worker_mem_gb"]].drop_duplicates()
    topology = snap_topology(topology_requested, topo_unique)
    print(f"Topology (snapped to dataset): {topology['topology_workers']} workers × {topology['topology_worker_cores']} cores × {topology['topology_worker_mem_gb']}GB")

    # Param grid
    param_grid = dataset.get_param_grid()
    print(f"[2/5] Parameter grid: {len(param_grid)} parameters")
    for name, values in list(param_grid.items())[:5]:
        print(f"      {name}: {len(values)} values")
    if len(param_grid) > 5:
        print(f"      ... and {len(param_grid) - 5} more")
    print()

    # Baseline (default config)
    # Baseline config
    baseline_time_data = None
    if args.baseline_from_topology:
        default_config, baseline_time_data = pick_baseline_from_topology(dataset, topology)
        if default_config is None:
            print("[WARN] No data rows for this topology; falling back to default config.")
            default_config = get_default_config(dataset)
            baseline_time_data = None
    else:
        default_config = get_default_config(dataset)

    default_time = predict_fn(pd.DataFrame([{**topology, "profile": args.profile, **default_config}]))[0]
    print(f"[3/5] Baseline: {default_time:.2f}s")
    if baseline_time_data is not None:
        print(f"      Dataset median for this topology: {baseline_time_data:.2f}s")
    print()

    # MLflow setup
    mlflow = None
    if args.mlflow:
        try:
            mlflow = setup_mlflow(args.mlflow_uri, args.experiment)
            print(f"[MLflow] Connected to {args.mlflow_uri}")
        except Exception as e:
            print(f"[WARN] MLflow connection failed: {e}")
            mlflow = None

    # Run optimization
    print("[4/5] Running optimization...")
    results = {}

    if args.algorithm in ["all", "qlearning"]:
        print("  [Q-Learning]")
        ql = TabularQLearning(
            param_grid, predict_fn,
            alpha=args.ql_alpha,
            gamma=args.ql_gamma,
            epsilon=args.ql_epsilon,
            random_state=args.seed,
            initial_config=default_config,  # Стартуем с известной хорошей конфигурации
        )
        results["QLearning"] = ql.optimize(topology, args.profile, n_episodes=args.ql_episodes)
        print(f"    Best time: {results['QLearning'].best_predicted_time:.2f}s")
        print(f"    Speedup: {results['QLearning'].speedup_vs_default(default_time):.2f}x")

    if args.algorithm in ["all", "dqn"]:
        try:
            print("  [DQN]")
            dqn = DQNOptimizer(param_grid, predict_fn, random_state=args.seed, initial_config=default_config)
            results["DQN"] = dqn.optimize(topology, args.profile, n_episodes=args.dqn_episodes)
            print(f"    Best time: {results['DQN'].best_predicted_time:.2f}s")
            print(f"    Speedup: {results['DQN'].speedup_vs_default(default_time):.2f}x")
        except ImportError as e:
            print(f"    Skipped: {e}")

    if args.algorithm in ["all", "bayesian"]:
        try:
            print("  [Bayesian]")
            bo = BayesianOptimizer(param_grid, predict_fn, random_state=args.seed)
            results["Bayesian"] = bo.optimize(topology, args.profile, n_trials=args.bo_trials)
            print(f"    Best time: {results['Bayesian'].best_predicted_time:.2f}s")
            print(f"    Speedup: {results['Bayesian'].speedup_vs_default(default_time):.2f}x")
        except ImportError as e:
            print(f"    Skipped: {e}")

    if args.algorithm in ["all", "ppo"]:
        try:
            print("  [PPO]")
            ppo = StableBaselinesOptimizer(param_grid, predict_fn, algorithm="PPO", random_state=args.seed)
            results["PPO"] = ppo.optimize(topology, args.profile, total_timesteps=args.ppo_timesteps)
            print(f"    Best time: {results['PPO'].best_predicted_time:.2f}s")
            print(f"    Speedup: {results['PPO'].speedup_vs_default(default_time):.2f}x")
        except ImportError as e:
            print(f"    Skipped: {e}")

    print()

    # Save results
    print("[5/5] Saving results...")

    # Report with ordered keys and cleaned numbers
    order_topology = ["topology_workers", "topology_worker_cores", "topology_worker_mem_gb"]
    order_params = [
        "executor_cores", "executor_instances",
        "driver_cores",
        "executor_memory_mb", "driver_memory_mb",
        "memory_fraction", "memory_storageFraction",
        "rpc_message_maxSize",
        "shuffle_file_buffer_kb", "broadcast_block_mb", "maxSizeInFlight_mb",
        "shuffle_compress", "spill_compress", "broadcast_compress", "rdd_compress",
        "io_codec",
    ]

    def order_dict(src: dict, keys: list[str]) -> dict:
        return {k: clean_number(src[k]) for k in keys if k in src}

    report = {
        "dataset": str(args.csv),
        "model_dir": str(args.model_dir),
        "predictor_type": args.predictor_type,
        "topology": order_dict(topology, order_topology),
        "profile": args.profile,
        "baseline_time_pred": float(default_time),
        "baseline_time_data": float(baseline_time_data) if baseline_time_data is not None else None,
        "default_config": order_dict(default_config, order_params),
        "results": {},
    }

    for name, result in results.items():
        # reorder best_config for readability (same order as params)
        best_cfg = order_dict(result.best_config, order_params)

        report["results"][name] = {
            "best_predicted_time": result.best_predicted_time,
            "speedup": result.speedup_vs_default(default_time),
            "improvement_pct": result.improvement_pct(default_time),
            "best_config": best_cfg,
            "n_iterations": result.n_iterations,
        }

        # Save history
        result.history.to_csv(outdir / f"history_{name.lower()}.csv", index=False)

        # Plot trajectory
        plot_optimization_trajectory(result, plots_dir)

    with open(outdir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"      Report saved to: {outdir / 'report.json'}")

    # Comparison plot
    if len(results) > 1:
        plot_comparison(results, default_time, plots_dir)
    print(f"      Plots saved to: {plots_dir}")

    # MLflow
    if mlflow and results:
        with mlflow.start_run(run_name="rl_optimization"):
            mlflow.log_params({
                "topology_workers": args.workers,
                "topology_worker_cores": args.worker_cores,
                "topology_worker_mem": args.worker_mem,
                "profile": args.profile,
                "baseline_time": default_time,
            })
            for name, result in results.items():
                mlflow.log_metric(f"{name}_best_time", result.best_predicted_time)
                mlflow.log_metric(f"{name}_speedup", result.speedup_vs_default(default_time))
            mlflow.log_artifact(str(outdir / "report.json"))
            for png in plots_dir.glob("*.png"):
                mlflow.log_artifact(str(png))
        print("      MLflow logging complete")

    # Summary
    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Baseline (default): {default_time:.2f}s")
    print()
    print(f"{'Algorithm':<15} {'Best Time':>12} {'Speedup':>10} {'Improvement':>12}")
    print("-" * 50)
    for name, result in sorted(results.items(), key=lambda x: x[1].best_predicted_time):
        speedup = result.speedup_vs_default(default_time)
        improvement = result.improvement_pct(default_time)
        print(f"{name:<15} {result.best_predicted_time:>10.2f}s {speedup:>9.2f}x {improvement:>10.1f}%")

    if results:
        best_algo = min(results.items(), key=lambda x: x[1].best_predicted_time)
        print()
        print(f"Best algorithm: {best_algo[0]}")
        print(f"Best config:")
        for k, v in best_algo[1].best_config.items():
            print(f"  {k}: {v}")

    print()
    print(f"[OK] All artifacts saved to: {outdir}")


if __name__ == "__main__":
    main()
