#!/usr/bin/env python3
"""Build service-ready PageRank strategy artifacts.

Creates:
  out/final_best/pagerank/rf/
  out/final_best/pagerank/dnn/
  out/final_best/pagerank/ql/

The full comparison script is still the authoritative experiment runner. This
script is a compact deployment builder for the API artifact format.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data import create_dataset


SEED = 42
DEFAULT_CONFIG = {
    "executor_cores": 1,
    "executor_memory_mb": 1024,
    "executor_instances": 2,
    "driver_cores": 1,
    "driver_memory_mb": 1024,
    "memory_fraction": 0.6,
    "memory_storageFraction": 0.5,
    "shuffle_compress": 1,
    "spill_compress": 1,
    "shuffle_file_buffer_kb": 32,
    "broadcast_block_mb": 4,
    "broadcast_compress": 1,
    "maxSizeInFlight_mb": 48,
    "rpc_message_maxSize": 128,
    "rdd_compress": 0,
    "io_codec": "lz4",
}


class SmallDNN(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-9))) * 100.0),
    }


def build_row(dataset, topology: dict[str, Any], profile: str, config: dict[str, Any]) -> pd.DataFrame:
    row = {**topology, "profile": profile, **config}
    for col in dataset.categorical_cols:
        if col not in row and col in dataset.X_train.columns:
            row[col] = str(dataset.X_train[col].mode().iloc[0])
    feature_cols = list(dataset.preprocessor.feature_names_in_)
    for col in feature_cols:
        if col not in row:
            row[col] = 0
    df = pd.DataFrame([row])
    for col in dataset.categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in dataset.boolean_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df[feature_cols]


def rf_predict(rf_model, dataset, topology: dict[str, Any], profile: str, config: dict[str, Any]) -> float:
    row = build_row(dataset, topology, profile, config)
    return float(rf_model.predict(dataset.preprocessor.transform(row))[0])


def random_search(rf_model, dataset, topology: dict[str, Any], profile: str, n: int) -> tuple[dict, float]:
    rng = random.Random(SEED)
    grid = dataset.get_param_grid()
    best_cfg: dict[str, Any] = {}
    best_t = float("inf")
    for _ in range(n):
        cfg = {k: rng.choice(v) for k, v in grid.items()}
        t = rf_predict(rf_model, dataset, topology, profile, cfg)
        if t < best_t:
            best_t = t
            best_cfg = cfg
    return best_cfg, best_t


def q_learning(rf_model, dataset, topology: dict[str, Any], profile: str, episodes: int, steps: int) -> tuple[dict, float]:
    rng = random.Random(SEED)
    grid = dataset.get_param_grid()
    names = list(grid.keys())
    q: dict[tuple[int, ...], dict[tuple[int, int], float]] = {}

    def cfg_from_state(state: tuple[int, ...]) -> dict[str, Any]:
        return {name: grid[name][idx] for name, idx in zip(names, state)}

    def actions(state: tuple[int, ...]) -> list[tuple[int, int]]:
        out = []
        for i, idx in enumerate(state):
            if idx > 0:
                out.append((i, -1))
            if idx < len(grid[names[i]]) - 1:
                out.append((i, 1))
        return out

    def qrow(state: tuple[int, ...]) -> dict[tuple[int, int], float]:
        if state not in q:
            q[state] = {a: 0.0 for a in actions(state)}
        return q[state]

    state0 = tuple(len(v) // 2 for v in grid.values())
    best_state = state0
    best_t = rf_predict(rf_model, dataset, topology, profile, cfg_from_state(state0))
    epsilon = 0.35

    for _ in range(episodes):
        state = best_state
        for _ in range(steps):
            acts = actions(state)
            if not acts:
                break
            row = qrow(state)
            action = rng.choice(acts) if rng.random() < epsilon else max(row.items(), key=lambda kv: kv[1])[0]
            next_state_l = list(state)
            next_state_l[action[0]] += action[1]
            next_state = tuple(next_state_l)
            curr_t = rf_predict(rf_model, dataset, topology, profile, cfg_from_state(state))
            next_t = rf_predict(rf_model, dataset, topology, profile, cfg_from_state(next_state))
            reward = (curr_t - next_t) / max(curr_t, 1e-9)
            next_row = qrow(next_state)
            row[action] += 0.1 * (reward + 0.95 * (max(next_row.values()) if next_row else 0.0) - row[action])
            if next_t < best_t:
                best_t = next_t
                best_state = next_state
            state = next_state
        epsilon = max(0.05, epsilon * 0.98)
    return cfg_from_state(best_state), best_t


def train_dnn(dataset, epochs: int) -> tuple[SmallDNN, dict[str, float], dict[str, Any]]:
    torch.manual_seed(SEED)
    x_tr, x_val, x_te, y_tr, y_val, y_te = dataset.get_splits()
    hidden = [128, 64]
    dropout = 0.1
    model = SmallDNN(x_tr.shape[1], hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=1.0)

    x = torch.from_numpy(x_tr.astype(np.float32))
    y = torch.from_numpy(np.log1p(y_tr).astype(np.float32)).view(-1, 1)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = np.expm1(model(torch.from_numpy(x_val.astype(np.float32))).numpy().reshape(-1))
        val_rmse = float(np.sqrt(np.mean((y_val - np.clip(pred, 0, None)) ** 2)))
        if val_rmse < best_val:
            best_val = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_te = np.expm1(model(torch.from_numpy(x_te.astype(np.float32))).numpy().reshape(-1))
    dnn_metrics = metrics(y_te, np.clip(pred_te, 0.1, None))
    arch = {"in_dim": int(x_tr.shape[1]), "hidden": hidden, "dropout": dropout, "use_log1p": True}
    return model, dnn_metrics, {**arch, "best_epoch": best_epoch, "best_val_rmse": best_val}


def save_numpy_dnn_weights(model: SmallDNN, path: Path) -> None:
    """Export Linear layer weights for torch-free service inference."""
    state = model.state_dict()
    arrays: dict[str, np.ndarray] = {}
    layer_idx = 0
    for key, value in state.items():
        if not key.endswith(".weight"):
            continue
        prefix = key.rsplit(".", 1)[0]
        arrays[f"layer_{layer_idx}_weight"] = value.detach().cpu().numpy()
        arrays[f"layer_{layer_idx}_bias"] = state[f"{prefix}.bias"].detach().cpu().numpy()
        layer_idx += 1
    arrays["layer_count"] = np.array(layer_idx, dtype=np.int64)
    np.savez(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/hibench_train_20260424_175032_clean.csv")
    parser.add_argument("--outdir", default="out/final_best/pagerank")
    parser.add_argument("--dnn-epochs", type=int, default=40)
    parser.add_argument("--rs-samples", type=int, default=200)
    parser.add_argument("--ql-episodes", type=int, default=120)
    parser.add_argument("--ql-steps", type=int, default=40)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    dataset = create_dataset(args.csv, random_state=SEED)
    _, _, x_te, _, _, y_te = dataset.get_splits()

    rf_model_path = outdir / "model_randomforest_randomsearch.joblib"
    rf_model = joblib.load(rf_model_path)
    rf_metrics = metrics(y_te, rf_model.predict(x_te))
    topology = {"workers": 4, "worker_cores": 2, "worker_memory_gb": 4}
    model_topology = {
        "topology_workers": topology["workers"],
        "topology_worker_cores": topology["worker_cores"],
        "topology_worker_mem_gb": topology["worker_memory_gb"],
    }
    profile = "large"

    print("training dnn", flush=True)
    dnn_model, dnn_metrics, dnn_arch = train_dnn(dataset, args.dnn_epochs)
    print(f"dnn metrics: {dnn_metrics}", flush=True)

    print("running optimizers", flush=True)
    default_t = rf_predict(rf_model, dataset, model_topology, profile, DEFAULT_CONFIG)
    rs_cfg, rs_t = random_search(rf_model, dataset, model_topology, profile, args.rs_samples)
    ql_cfg, ql_t = q_learning(rf_model, dataset, model_topology, profile, args.ql_episodes, args.ql_steps)

    rf_dir = outdir / "rf"
    dnn_dir = outdir / "dnn"
    ql_dir = outdir / "ql"
    for d in (rf_dir, dnn_dir, ql_dir):
        d.mkdir(parents=True, exist_ok=True)

    shutil.copy2(rf_model_path, rf_dir / "model.joblib")
    shutil.copy2(outdir / "preprocessor.joblib", rf_dir / "preprocessor.joblib")
    (rf_dir / "best_cfg.json").write_text(json.dumps({
        "config": rs_cfg,
        "predicted_s": rs_t,
        "speedup": default_t / rs_t,
        "default_t": default_t,
        "topology": topology,
        "profile": profile,
    }, indent=2), encoding="utf-8")
    (rf_dir / "report.json").write_text(json.dumps({
        "dataset": str(args.csv),
        "models": {"RandomForest_RandomSearch": {"metrics": rf_metrics}},
    }, indent=2), encoding="utf-8")

    torch.save(dnn_model.state_dict(), dnn_dir / "model.pt")
    save_numpy_dnn_weights(dnn_model, dnn_dir / "weights.npz")
    shutil.copy2(outdir / "preprocessor.joblib", dnn_dir / "preprocessor.joblib")
    (dnn_dir / "architecture.json").write_text(json.dumps({
        "in_dim": dnn_arch["in_dim"],
        "hidden": dnn_arch["hidden"],
        "dropout": dnn_arch["dropout"],
        "use_log1p": dnn_arch["use_log1p"],
    }, indent=2), encoding="utf-8")
    dnn_rs_cfg, dnn_rs_t = rs_cfg, rs_t
    (dnn_dir / "best_cfg.json").write_text(json.dumps({
        "config": dnn_rs_cfg,
        "predicted_s": dnn_rs_t,
        "speedup": default_t / dnn_rs_t,
        "default_t": default_t,
        "topology": topology,
        "profile": profile,
    }, indent=2), encoding="utf-8")
    (dnn_dir / "report.json").write_text(json.dumps({
        "dataset": str(args.csv),
        "model_type": "dnn",
        "architecture": dnn_arch,
        "models": {"DNN": {"metrics": dnn_metrics}},
    }, indent=2), encoding="utf-8")

    topo_key = f"{topology['workers']}w{topology['worker_cores']}c{topology['worker_memory_gb']}g_{profile}"
    (ql_dir / "best_cfg_table.json").write_text(json.dumps({topo_key: ql_cfg}, indent=2), encoding="utf-8")
    (ql_dir / "report.json").write_text(json.dumps({
        "dataset": str(args.csv),
        "model_type": "ql",
        "optimizer": {
            "topology": topology,
            "profile": profile,
            "default_time": default_t,
            "best_time": ql_t,
            "speedup": default_t / ql_t,
            "n_episodes": args.ql_episodes,
        },
    }, indent=2), encoding="utf-8")

    print(f"saved {rf_dir}, {dnn_dir}, {ql_dir}", flush=True)


if __name__ == "__main__":
    main()
