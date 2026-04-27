#!/usr/bin/env python3
"""
Сравнение трёх подходов к предсказанию времени выполнения Spark:
  Model 1 — RandomForest + RandomSearch   (production baseline)
  Model 2 — DNN predictor                 (архитектура Sensors-22, улучшенный)
  Model 3 — RF surrogate + Q-learning     (Sensors-22 RL-парадигма)

Запуск:
    python training/train_compare_models.py \
        --pagerank-csv data/hibench_train_20260424_175032_clean.csv \
        --wordcount-csv data/wc_train_merged.csv \
        --outdir out/model_comparison
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data import create_dataset, SparkDataset
from models.baseline import compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG_SEED = 42
_REPORT_JSON = "report.json"


from dataclasses import dataclass as _dc


@_dc
class _ArtifactData:
    rf_model: Any
    dnn_model: Any
    dnn_cfg_info: Dict
    dataset: Any
    rf_metrics: Dict
    dnn_metrics: Dict
    rs_best_cfg: Dict
    rs_best_t: float
    dnn_rs_best_cfg: Dict
    dnn_rs_best_t: float
    ql_best_cfg: Dict
    ql_best_t: float
    default_t: float
    topology: Dict
    profile: str
    n_features: int


# ── DNN model ────────────────────────────────────────────────────────────────

class _DNN(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...], dropout: float, use_bn: bool):
        super().__init__()
        layers: list = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_dnn_torch(
    x_tr: np.ndarray, y_tr: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    hidden: Tuple[int, ...] = (128, 64),
    dropout: float = 0.2,
    use_bn: bool = False,
    lr: float = 1e-3,
    batch: int = 32,
    max_epochs: int = 800,
    patience: int = 80,
) -> Tuple[_DNN, Dict[str, list], int]:
    """Trains DNN; returns (model, history_dict, best_epoch)."""
    torch.manual_seed(RNG_SEED)
    model = _DNN(x_tr.shape[1], hidden, dropout, use_bn).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Cosine annealing: lr decays smoothly — avoids premature convergence
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=lr * 0.01)
    loss_fn = nn.HuberLoss(delta=1.0)  # robust to outliers vs pure MSE

    y_tr_log = np.log1p(y_tr).astype(np.float32)

    ds = TensorDataset(
        torch.from_numpy(x_tr.astype(np.float32)),
        torch.from_numpy(y_tr_log).view(-1, 1),
    )
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=len(ds) > batch,
                        num_workers=0)
    x_val_t = torch.from_numpy(x_val.astype(np.float32)).to(DEVICE)

    hist_train, hist_val_rmse = [], []
    best_val_rmse = float("inf")
    best_state, best_epoch = None, 0
    patience_ctr = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(ds)
        sched.step()

        model.eval()
        with torch.no_grad():
            y_pred_log = model(x_val_t).cpu().numpy().reshape(-1)
        y_pred = np.clip(np.expm1(y_pred_log), 0, None)
        val_rmse = float(np.sqrt(np.mean((y_val - y_pred) ** 2)))

        hist_train.append(float(ep_loss))
        hist_val_rmse.append(val_rmse)

        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"train_loss": hist_train, "val_rmse": hist_val_rmse}, best_epoch


def _dnn_predict(model: _DNN, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
        y_log = model(t).cpu().numpy().reshape(-1)
    return np.clip(np.expm1(y_log), 0, None)


# DNN search space: no BN (hurts on ~600 samples), vary depth/width/regularisation
_DNN_SEARCH = [
    # Exact Sensors-22 architecture
    {"hidden": (128, 64),       "dropout": 0.1,  "use_bn": False, "lr": 1e-3},
    # Wider — retains Sensors-22 structure but more capacity
    {"hidden": (256, 128),      "dropout": 0.15, "use_bn": False, "lr": 1e-3},
    # Deeper — added layer + stronger regularisation
    {"hidden": (256, 128, 64),  "dropout": 0.2,  "use_bn": False, "lr": 8e-4},
    # Slower lr, exact architecture
    {"hidden": (128, 64),       "dropout": 0.2,  "use_bn": False, "lr": 3e-4},
    # Even wider, moderate dropout
    {"hidden": (512, 256, 128), "dropout": 0.25, "use_bn": False, "lr": 5e-4},
]


def train_best_dnn(
    x_tr: np.ndarray, y_tr: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    verbose: bool = True,
) -> Tuple[_DNN, Dict, Dict]:
    """Tries several DNN configs; returns (best_model, best_cfg_info, best_history)."""
    best_rmse = float("inf")
    best_model, best_cfg_info, best_hist = None, None, None

    for cfg in _DNN_SEARCH:
        if verbose:
            print(f"  DNN {cfg['hidden']} do={cfg['dropout']} lr={cfg['lr']}")
        m, hist, ep = _train_dnn_torch(x_tr, y_tr, x_val, y_val, **cfg)
        val_rmse = float(min(hist["val_rmse"]))
        if verbose:
            print(f"    best val RMSE={val_rmse:.3f}s @ epoch {ep}")
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model = m
            best_cfg_info = {**cfg, "best_val_rmse": val_rmse, "best_epoch": ep}
            best_hist = hist

    return best_model, best_cfg_info, best_hist


# ── RF + RandomSearch ─────────────────────────────────────────────────────────

def train_rf_rs(
    x_tr: np.ndarray, y_tr: np.ndarray,
    n_iter: int = 30,
    verbose: bool = True,
) -> Tuple[RandomForestRegressor, Dict]:
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 15, 20],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2],
        "max_features": [None, "sqrt", 0.5],
    }
    # n_jobs=1 everywhere: WSL multiprocessing fork causes hangs with n_jobs=-1
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=RNG_SEED, n_jobs=1),
        param_dist, n_iter=n_iter, cv=3,
        scoring="neg_mean_absolute_error",
        random_state=RNG_SEED, n_jobs=1,
    )
    search.fit(x_tr, y_tr)
    if verbose:
        print(f"  RF best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


# ── Row building helpers ──────────────────────────────────────────────────────

def _build_row(
    dataset: SparkDataset,
    topology_workers: int, topology_worker_cores: int, topology_worker_mem_gb: int,
    profile: str,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build a properly typed single-row DataFrame for the dataset's preprocessor.
    Fills any missing categorical column using its mode from training data.
    """
    base = {
        "topology_workers": topology_workers,
        "topology_worker_cores": topology_worker_cores,
        "topology_worker_mem_gb": topology_worker_mem_gb,
        "profile": profile,
    }
    base.update(config)

    # Fill missing categorical columns (e.g. job_type for PageRank dataset)
    for col in dataset.categorical_cols:
        if col not in base and col in dataset.X_train.columns:
            base[col] = str(dataset.X_train[col].mode().iloc[0])

    feature_cols = dataset.numeric_cols + dataset.categorical_cols + dataset.boolean_cols
    df = pd.DataFrame([base])
    # Ensure categoricals are object dtype to avoid OHE isnan bug in sklearn 1.8
    for col in dataset.categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in dataset.boolean_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    # Fill missing columns with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]


# ── Spark default config (processed column names) ────────────────────────────

_SPARK_DEFAULTS = {
    "executor_cores": 1,
    "executor_memory_mb": 1024.0,
    "executor_instances": 2,
    "driver_cores": 1,
    "driver_memory_mb": 1024.0,
    "memory_fraction": 0.6,
    "memory_storageFraction": 0.5,
    "shuffle_compress": 1,
    "spill_compress": 1,
    "shuffle_file_buffer_kb": 32.0,
    "broadcast_block_mb": 4.0,
    "broadcast_compress": 1,
    "maxSizeInFlight_mb": 48.0,
    "rpc_message_maxSize": 128,
    "rdd_compress": 0,
    "io_codec": "lz4",
}


def _rf_predict(rf: RandomForestRegressor, df: pd.DataFrame, dataset: SparkDataset) -> float:
    X = dataset.preprocessor.transform(df)
    return float(rf.predict(X)[0])


def default_runtime(rf, dataset, topo, profile) -> float:
    df = _build_row(dataset, topo["workers"], topo["worker_cores"], topo["worker_memory_gb"],
                    profile, _SPARK_DEFAULTS)
    return _rf_predict(rf, df, dataset)


# ── Q-learning with RF surrogate ─────────────────────────────────────────────

def _ql_pred_cfg(
    rf: RandomForestRegressor,
    dataset: SparkDataset,
    topo_mapped: Dict[str, Any],
    profile: str,
    cfg: Dict[str, Any],
) -> float:
    """Predict runtime for one config dict using the RF surrogate."""
    row_df = _build_row(
        dataset,
        topo_mapped["topology_workers"],
        topo_mapped["topology_worker_cores"],
        topo_mapped["topology_worker_mem_gb"],
        profile, cfg,
    )
    return _rf_predict(rf, row_df, dataset)


def _ql_update(optimizer, state, action, next_state, reward, alpha, gamma) -> None:
    """Single Q-table update (Bellman equation)."""
    q_row = optimizer._get_q_row(state)
    if action not in q_row:
        return
    next_q_row = optimizer._get_q_row(next_state)
    max_next = max(next_q_row.values()) if next_q_row else 0.0
    q_row[action] += alpha * (reward + gamma * max_next - q_row[action])


def _ql_run_episode(
    optimizer, rf, dataset, topo_mapped, profile, param_grid,
    n_steps, alpha, gamma, eps, rng, ep_best_time, ep_best_cfg,
):
    """Run one Q-learning episode; returns updated (ep_best_time, ep_best_cfg)."""
    state_cfg = {name: rng.choice(vals) for name, vals in param_grid.items()}
    state = optimizer._state_to_indices(state_cfg)

    for _ in range(n_steps):
        action = optimizer._select_action(state, eps)
        if action is None:
            break
        next_state = optimizer._apply_action(state, action)
        curr_cfg = optimizer._indices_to_config(state)
        next_cfg = optimizer._indices_to_config(next_state)

        t_curr = _ql_pred_cfg(rf, dataset, topo_mapped, profile, curr_cfg)
        t_next = _ql_pred_cfg(rf, dataset, topo_mapped, profile, next_cfg)
        reward = (t_curr - t_next) / max(t_curr, 1e-6)

        _ql_update(optimizer, state, action, next_state, reward, alpha, gamma)

        if t_next < ep_best_time:
            ep_best_time = t_next
            ep_best_cfg = next_cfg.copy()

        state = next_state

    return ep_best_time, ep_best_cfg


def run_qlearning(
    rf: RandomForestRegressor,
    dataset: SparkDataset,
    topology: Dict[str, Any],
    profile: str,
    n_episodes: int = 800,
    n_steps: int = 60,
    alpha: float = 0.15,
    gamma: float = 0.95,
    epsilon: float = 0.40,
    eps_decay: float = 0.993,
    eps_min: float = 0.05,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], float, Dict[str, list]]:
    """Offline Q-learning with RF surrogate. Returns (best_cfg, best_time, history)."""
    from models.rl_optimizer import TabularQLearning

    param_grid = dataset.get_param_grid()
    optimizer = TabularQLearning(
        param_grid=param_grid,
        predictor=lambda df: np.zeros(len(df)),  # unused — we call _ql_pred_cfg directly
        alpha=alpha, gamma=gamma,
        epsilon=epsilon, epsilon_decay=eps_decay, epsilon_min=eps_min,
        random_state=RNG_SEED,
    )

    topo_mapped = {
        "topology_workers": topology["workers"],
        "topology_worker_cores": topology["worker_cores"],
        "topology_worker_mem_gb": topology["worker_memory_gb"],
    }

    best_time_per_ep: list = []
    ep_best_cfg: dict = {}
    ep_best_time = float("inf")
    rng = random.Random(RNG_SEED)
    eps = epsilon

    for ep in range(1, n_episodes + 1):
        ep_best_time, ep_best_cfg = _ql_run_episode(
            optimizer, rf, dataset, topo_mapped, profile, param_grid,
            n_steps, alpha, gamma, eps, rng, ep_best_time, ep_best_cfg,
        )
        best_time_per_ep.append(ep_best_time)
        eps = max(eps_min, eps * eps_decay)

        if verbose and ep % 100 == 0:
            print(f"  QL ep {ep}/{n_episodes}  best={ep_best_time:.2f}s  eps={eps:.3f}")

    return ep_best_cfg, ep_best_time, {"best_time_per_ep": best_time_per_ep}


def run_random_search_optimizer(
    rf: RandomForestRegressor,
    dataset: SparkDataset,
    topology: Dict[str, Any],
    profile: str,
    n_samples: int = 400,
) -> Tuple[Dict[str, Any], float]:
    param_grid = dataset.get_param_grid()
    rng = random.Random(RNG_SEED + 1)
    best_t, best_cfg = float("inf"), {}

    for _ in range(n_samples):
        cfg = {name: rng.choice(vals) for name, vals in param_grid.items()}
        row_df = _build_row(dataset,
            topology["workers"], topology["worker_cores"], topology["worker_memory_gb"],
            profile, cfg)
        t = _rf_predict(rf, row_df, dataset)
        if t < best_t:
            best_t, best_cfg = t, cfg.copy()

    return best_cfg, best_t


def run_random_search_optimizer_dnn(
    dnn_model: _DNN,
    dataset: SparkDataset,
    topology: Dict[str, Any],
    profile: str,
    n_samples: int = 400,
) -> Tuple[Dict[str, Any], float]:
    """Random search over config space using DNN as surrogate predictor."""
    param_grid = dataset.get_param_grid()
    rng = random.Random(RNG_SEED + 2)
    best_t, best_cfg = float("inf"), {}

    for _ in range(n_samples):
        cfg = {name: rng.choice(vals) for name, vals in param_grid.items()}
        row_df = _build_row(dataset,
            topology["workers"], topology["worker_cores"], topology["worker_memory_gb"],
            profile, cfg)
        X = dataset.preprocessor.transform(row_df)
        t = float(_dnn_predict(dnn_model, X)[0])
        if t < best_t:
            best_t, best_cfg = t, cfg.copy()

    return best_cfg, best_t


def save_artifacts(artifacts_dir: Path, d: "_ArtifactData") -> None:
    """Save RF, DNN and QL artifacts for multi-model service integration."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ── RF ──────────────────────────────────────────────────────────────────
    rf_dir = artifacts_dir / "rf"
    rf_dir.mkdir(exist_ok=True)
    joblib.dump(d.rf_model, rf_dir / "model.joblib")
    joblib.dump(d.dataset.preprocessor, rf_dir / "preprocessor.joblib")
    rf_speedup = d.default_t / d.rs_best_t if d.rs_best_t > 0 else 1.0
    with open(rf_dir / "best_cfg.json", "w") as f:
        json.dump({"config": d.rs_best_cfg, "predicted_s": float(d.rs_best_t),
                   "speedup": float(rf_speedup), "default_t": float(d.default_t),
                   "topology": d.topology, "profile": d.profile}, f, indent=2)
    with open(rf_dir / _REPORT_JSON, "w") as f:
        json.dump({
            "dataset": "pagerank_hibench_2026",
            "models": {"RF_RandomSearch": {"metrics": {k: float(v) for k, v in d.rf_metrics.items()}}},
        }, f, indent=2)

    # ── DNN ─────────────────────────────────────────────────────────────────
    dnn_dir = artifacts_dir / "dnn"
    dnn_dir.mkdir(exist_ok=True)
    torch.save(d.dnn_model.state_dict(), dnn_dir / "model.pt")
    arch = {
        "in_dim": d.n_features,
        "hidden": [int(h) for h in d.dnn_cfg_info.get("hidden", [])],
        "dropout": float(d.dnn_cfg_info.get("dropout", 0.0)),
        "use_log1p": True,
    }
    with open(dnn_dir / "architecture.json", "w") as f:
        json.dump(arch, f, indent=2)
    joblib.dump(d.dataset.preprocessor, dnn_dir / "preprocessor.joblib")
    dnn_speedup = d.default_t / d.dnn_rs_best_t if d.dnn_rs_best_t > 0 else 1.0
    with open(dnn_dir / "best_cfg.json", "w") as f:
        json.dump({"config": d.dnn_rs_best_cfg, "predicted_s": float(d.dnn_rs_best_t),
                   "speedup": float(dnn_speedup), "default_t": float(d.default_t),
                   "topology": d.topology, "profile": d.profile}, f, indent=2)
    with open(dnn_dir / _REPORT_JSON, "w") as f:
        json.dump({
            "dataset": "pagerank_hibench_2026",
            "model_type": "dnn",
            "architecture": arch,
            "models": {"DNN": {"metrics": {k: float(v) for k, v in d.dnn_metrics.items()}}},
        }, f, indent=2)

    # ── QL ──────────────────────────────────────────────────────────────────
    ql_dir = artifacts_dir / "ql"
    ql_dir.mkdir(exist_ok=True)
    topo_key = (f"{d.topology['workers']}w{d.topology['worker_cores']}"
                f"c{d.topology['worker_memory_gb']}g_{d.profile}")
    ql_speedup = d.default_t / d.ql_best_t if d.ql_best_t > 0 else 1.0
    with open(ql_dir / "best_cfg_table.json", "w") as f:
        json.dump({topo_key: d.ql_best_cfg}, f, indent=2)
    with open(ql_dir / _REPORT_JSON, "w") as f:
        json.dump({
            "dataset": "pagerank_hibench_2026",
            "model_type": "ql",
            "optimizer": {
                "topology": d.topology, "profile": d.profile,
                "default_time": float(d.default_t),
                "best_time": float(d.ql_best_t),
                "speedup": float(ql_speedup),
            },
        }, f, indent=2)

    print(f"[OK] Artifacts saved → {artifacts_dir}/{{rf,dnn,ql}}/")



# ── Plots ─────────────────────────────────────────────────────────────────────

def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metrics_bar(metrics_table: Dict[str, Dict[str, float]], outdir: Path, tag: str):
    models = list(metrics_table.keys())
    for metric in ["MAE", "MAPE", "R2", "RMSE"]:
        vals = [metrics_table[m].get(metric, 0) for m in models]
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#4C72B0", "#DD8452", "#55A868"][:len(models)]
        bars = ax.bar(models, vals, color=colors, edgecolor="black", width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.015,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        if metric == "MAPE":
            unit = "%"
        elif metric in ("MAE", "RMSE"):
            unit = "s"
        else:
            unit = ""
        ax.set_ylabel(f"{metric} {unit}")
        ax.set_title(f"{tag} — {metric} comparison")
        ax.set_ylim(0, max(vals) * 1.25)
        _save(fig, outdir / f"{tag.lower().replace(' ', '_')}_{metric.lower()}_bar.png")


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.45, s=18, c="#4C72B0")
    lims = [min(y_true.min(), y_pred.min()) * 0.95, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual (s)")
    ax.set_ylabel("Predicted (s)")
    ax.set_title(title)
    _save(fig, path)


def plot_dnn_curves(history: Dict[str, list], best_epoch: int, title: str, path: Path):
    val_rmse = history["val_rmse"]
    train_loss = history["train_loss"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_loss, alpha=0.8, color="#4C72B0", label="Train Loss (Huber log-space)")
    ax1.axvline(best_epoch - 1, color="red", linestyle="--", alpha=0.7, label=f"Best epoch {best_epoch}")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Huber Loss"); ax1.set_title(f"{title} — Train Loss"); ax1.legend()

    ax2.plot(val_rmse, alpha=0.8, color="#DD8452", label="Val RMSE (s)")
    ax2.axvline(best_epoch - 1, color="red", linestyle="--", alpha=0.7)
    ax2.axhline(min(val_rmse), color="green", linestyle=":", alpha=0.7,
                label=f"Best {min(val_rmse):.2f}s @ ep {best_epoch}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("RMSE (s)"); ax2.set_title(f"{title} — Val RMSE"); ax2.legend()
    _save(fig, path)


def plot_ql_convergence(history: Dict[str, list], rs_time: float, default_t: float,
                        title: str, path: Path):
    best_times = history["best_time_per_ep"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(best_times, color="#55A868", linewidth=1.2, label="Q-Learning (running best)")
    ax.axhline(rs_time, color="#DD8452", linestyle="--", linewidth=1.5,
               label=f"RandomSearch best {rs_time:.1f}s")
    ax.axhline(default_t, color="gray", linestyle=":", linewidth=1.5,
               label=f"Default config {default_t:.1f}s")
    ax.set_xlabel("Episode"); ax.set_ylabel("Best predicted time (s)"); ax.set_title(title); ax.legend()
    _save(fig, path)


def plot_optimizer_comparison(results: List[Dict], outdir: Path, tag: str):
    labels = [r["method"] for r in results]
    speedups = [r["speedup"] for r in results]
    times = [r["best_time"] for r in results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#4C72B0", "#55A868"]
    for ax, vals, ylabel, suffix in [
        (ax1, speedups, "Speedup ×", "Speedup vs default"),
        (ax2, times, "Predicted time (s)", "Best predicted runtime"),
    ]:
        bars = ax.bar(labels, vals, color=colors[:len(labels)], edgecolor="black", width=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.015,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(f"{tag} — {suffix}")
    _save(fig, outdir / f"{tag.lower().replace(' ', '_')}_optimizer_comparison.png")


def plot_cross_dataset(
    pr_metrics: Dict[str, Dict[str, float]],
    wc_metrics: Dict[str, Dict[str, float]],
    outdir: Path,
):
    models = list(pr_metrics.keys())
    x = np.arange(len(models))
    w = 0.35
    for metric in ["MAE", "MAPE"]:
        pr_vals = [pr_metrics[m].get(metric, 0) for m in models]
        wc_vals = [wc_metrics[m].get(metric, 0) for m in models]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - w / 2, pr_vals, w, label="PageRank", color="#4C72B0", edgecolor="black")
        ax.bar(x + w / 2, wc_vals, w, label="WordCount", color="#DD8452", edgecolor="black")
        ymax = max(pr_vals + wc_vals)
        for xi, (pv, wv) in enumerate(zip(pr_vals, wc_vals)):
            ax.text(xi - w / 2, pv + ymax * 0.015, f"{pv:.2f}", ha="center", va="bottom", fontsize=8)
            ax.text(xi + w / 2, wv + ymax * 0.015, f"{wv:.2f}", ha="center", va="bottom", fontsize=8)
        unit = "%" if metric == "MAPE" else "s"
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=12, ha="right")
        ax.set_ylabel(f"{metric} {unit}"); ax.set_title(f"Cross-dataset comparison — {metric}"); ax.legend()
        _save(fig, outdir / f"cross_dataset_{metric.lower()}.png")


# ── Per-dataset training pipeline ────────────────────────────────────────────

def run_dataset(
    csv_path: Path,
    dataset_label: str,
    outdir: Path,
    ref_topology: Dict,
    ref_profile: str,
    verbose: bool = True,
    save_dir: Optional[Path] = None,
) -> Dict:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_label}  ({csv_path.name})")
    print(f"{'='*60}")

    plots_dir = outdir / dataset_label.lower().replace(" ", "_")
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset = create_dataset(str(csv_path), random_state=RNG_SEED)
    x_tr, x_val, x_te, y_tr, y_val, y_te = dataset.get_splits()
    print(f"  train={len(y_tr)}  val={len(y_val)}  test={len(y_te)}  features={x_tr.shape[1]}")

    # ── Model 1: RF + RandomSearch ─────────────────────────────────────────
    print("\n[Model 1] RF + RandomSearch")
    rf_model, rf_params = train_rf_rs(x_tr, y_tr, verbose=verbose)
    rf_pred_te = rf_model.predict(x_te)
    rf_metrics = compute_metrics(y_te, rf_pred_te)
    print(f"  Test → MAE={rf_metrics['MAE']:.3f}s  MAPE={rf_metrics['MAPE']:.2f}%  R²={rf_metrics['R2']:.4f}")
    plot_scatter(y_te, rf_pred_te, f"RF+RS — {dataset_label}", plots_dir / "scatter_rf.png")

    # ── Model 2: DNN ──────────────────────────────────────────────────────
    print("\n[Model 2] DNN (config search, 5 architectures)")
    dnn_model, dnn_cfg, dnn_hist = train_best_dnn(x_tr, y_tr, x_val, y_val, verbose=verbose)
    dnn_pred_te = _dnn_predict(dnn_model, x_te)
    dnn_metrics = compute_metrics(y_te, dnn_pred_te)
    print(f"  Best arch: {dnn_cfg.get('hidden')}  do={dnn_cfg.get('dropout')}  lr={dnn_cfg.get('lr')}")
    print(f"  Test → MAE={dnn_metrics['MAE']:.3f}s  MAPE={dnn_metrics['MAPE']:.2f}%  R²={dnn_metrics['R2']:.4f}")
    plot_scatter(y_te, dnn_pred_te, f"DNN — {dataset_label}", plots_dir / "scatter_dnn.png")
    plot_dnn_curves(dnn_hist, dnn_cfg["best_epoch"], f"DNN — {dataset_label}",
                    plots_dir / "dnn_learning_curves.png")

    plot_metrics_bar({"RF+RS": rf_metrics, "DNN": dnn_metrics}, plots_dir, tag=dataset_label)

    # ── Model 3: RF surrogate + Q-learning ────────────────────────────────
    print("\n[Model 3] RF surrogate + Q-learning")
    default_t = default_runtime(rf_model, dataset, ref_topology, ref_profile)
    print(f"  Default config time (RF pred): {default_t:.2f}s")

    rs_best_cfg, rs_best_t = run_random_search_optimizer(
        rf_model, dataset, ref_topology, ref_profile, n_samples=400)
    rs_speedup = default_t / rs_best_t if rs_best_t > 0 else 0
    print(f"  RandomSearch: best={rs_best_t:.2f}s  speedup=×{rs_speedup:.3f}")

    ql_best_cfg, ql_best_t, ql_hist = run_qlearning(
        rf_model, dataset, ref_topology, ref_profile,
        n_episodes=800, verbose=verbose)
    ql_speedup = default_t / ql_best_t if ql_best_t > 0 else 0
    print(f"  Q-learning:   best={ql_best_t:.2f}s  speedup=×{ql_speedup:.3f}")

    dnn_rs_best_cfg, dnn_rs_best_t = run_random_search_optimizer_dnn(
        dnn_model, dataset, ref_topology, ref_profile, n_samples=400)
    dnn_rs_speedup = default_t / dnn_rs_best_t if dnn_rs_best_t > 0 else 0
    print(f"  DNN+RS:       best={dnn_rs_best_t:.2f}s  speedup=×{dnn_rs_speedup:.3f}")

    plot_ql_convergence(ql_hist, rs_best_t, default_t,
                        f"Q-Learning convergence — {dataset_label}",
                        plots_dir / "ql_convergence.png")
    plot_optimizer_comparison(
        [{"method": "RandomSearch", "speedup": rs_speedup, "best_time": rs_best_t},
         {"method": "Q-Learning",  "speedup": ql_speedup, "best_time": ql_best_t}],
        plots_dir, tag=dataset_label)

    if save_dir is not None:
        save_artifacts(save_dir, _ArtifactData(
            rf_model=rf_model, dnn_model=dnn_model, dnn_cfg_info=dnn_cfg,
            dataset=dataset, rf_metrics=rf_metrics, dnn_metrics=dnn_metrics,
            rs_best_cfg=rs_best_cfg, rs_best_t=float(rs_best_t),
            dnn_rs_best_cfg=dnn_rs_best_cfg, dnn_rs_best_t=float(dnn_rs_best_t),
            ql_best_cfg=ql_best_cfg, ql_best_t=float(ql_best_t),
            default_t=float(default_t), topology=ref_topology, profile=ref_profile,
            n_features=int(x_tr.shape[1]),
        ))

    return {
        "dataset": dataset_label,
        "n_train": int(len(y_tr)), "n_val": int(len(y_val)), "n_test": int(len(y_te)),
        "n_features": int(x_tr.shape[1]),
        "models": {
            "RF_RandomSearch": {
                "metrics": {k: float(v) for k, v in rf_metrics.items()},
                "best_params": {k: (None if v is None else str(v)) for k, v in rf_params.items()},
            },
            "DNN": {
                "metrics": {k: float(v) for k, v in dnn_metrics.items()},
                "config": {
                    "hidden": list(dnn_cfg.get("hidden", [])),
                    "dropout": float(dnn_cfg.get("dropout", 0)),
                    "lr": float(dnn_cfg.get("lr", 0)),
                    "best_epoch": int(dnn_cfg.get("best_epoch", 0)),
                    "best_val_rmse": float(dnn_cfg.get("best_val_rmse", 0)),
                },
            },
        },
        "optimizer": {
            "topology": ref_topology,
            "profile": ref_profile,
            "default_time": float(default_t),
            "random_search": {"best_time": float(rs_best_t), "speedup": float(rs_speedup)},
            "q_learning":    {"best_time": float(ql_best_t), "speedup": float(ql_speedup),
                              "n_episodes": 800},
        },
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pagerank-csv", required=True)
    parser.add_argument("--wordcount-csv", required=True)
    parser.add_argument("--outdir", default="out/model_comparison")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save-artifacts", action="store_true",
                        help="Save RF/DNN/QL artifacts for service integration (PageRank only)")
    parser.add_argument("--artifacts-dir", default="out/final_best/pagerank",
                        help="Directory for saved model artifacts (default: out/final_best/pagerank)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    ref_topology = {"workers": 4, "worker_cores": 2, "worker_memory_gb": 4}
    pr_save_dir = Path(args.artifacts_dir) if args.save_artifacts else None

    pr_result = run_dataset(
        Path(args.pagerank_csv), "PageRank", outdir,
        ref_topology, "large", verbose=verbose, save_dir=pr_save_dir)
    wc_result = run_dataset(
        Path(args.wordcount_csv), "WordCount", outdir,
        ref_topology, "large", verbose=verbose)

    plot_cross_dataset(pr_result["models"], wc_result["models"], outdir)

    report = {"pagerank": pr_result, "wordcount": wc_result}
    report_path = outdir / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Report → {report_path}")

    print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
    for ds_name, result in [("PageRank", pr_result), ("WordCount", wc_result)]:
        print(f"\n{ds_name}:")
        for mname, info in result["models"].items():
            m = info["metrics"]
            print(f"  {mname:<22} MAE={m['MAE']:.2f}s  MAPE={m['MAPE']:.1f}%  R²={m['R2']:.4f}")
        opt = result["optimizer"]
        print(f"  {'Optimizer RS':<22} best={opt['random_search']['best_time']:.1f}s  speedup=×{opt['random_search']['speedup']:.3f}")
        print(f"  {'Optimizer QL':<22} best={opt['q_learning']['best_time']:.1f}s  speedup=×{opt['q_learning']['speedup']:.3f}")


if __name__ == "__main__":
    main()
