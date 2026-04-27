"""
PredictorService: обёртка над sklearn-моделью.
DnnPredictorService: обёртка над PyTorch DNN (загружается из .pt + architecture.json).
QlLookupService: возвращает предвычисленные QL-конфиги, использует RF для оценки времени.

Принимает словари с конфигурацией, возвращает предсказание и
доверительную полосу по деревьям RF (p5/p95).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model_registry import ModelArtifact

# Spark-конфигурация по умолчанию для расчёта speedup
# Значения взяты из таблицы дефолтов Spark (Sensors-22, Table 1)
_SPARK_DEFAULTS = {
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


def _conf_band(rf_model, x_transformed: np.ndarray) -> tuple[float, float]:
    """Доверительная полоса: p5/p95 предсказаний по отдельным деревьям RF."""
    if not hasattr(rf_model, "estimators_"):
        pred = float(rf_model.predict(x_transformed)[0])
        return pred, pred
    tree_preds = np.array([t.predict(x_transformed)[0] for t in rf_model.estimators_])
    return float(np.percentile(tree_preds, 5)), float(np.percentile(tree_preds, 95))


class PredictorService:
    def __init__(self, artifact: ModelArtifact):
        self._model = artifact.predictor
        self._pre = artifact.preprocessor
        self._job_type = artifact.job_type
        self._feature_cols: list[str] = list(self._pre.feature_names_in_)

    def _enrich(self, row: dict) -> dict:
        """Добавляет job_type если препроцессор его ожидает."""
        if "job_type" in self._feature_cols and "job_type" not in row:
            return {**row, "job_type": self._job_type}
        return row

    def _to_df(self, row: dict) -> pd.DataFrame:
        return pd.DataFrame([self._enrich(row)])[self._feature_cols]

    def predict_one(self, row: dict) -> tuple[float, float, float]:
        """Возвращает (predicted_s, band_low, band_high)."""
        df = self._to_df(row)
        X = self._pre.transform(df)
        pred = float(self._model.predict(X)[0])
        pred = max(0.1, pred)
        low, high = _conf_band(self._model, X)
        return pred, max(0.1, low), max(0.1, high)

    def predict_batch(self, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Векторный вариант. Возвращает (preds, lows, highs)."""
        enriched = [self._enrich(r) for r in rows]
        df = pd.DataFrame(enriched)[self._feature_cols]
        X = self._pre.transform(df)
        preds = np.maximum(0.1, self._model.predict(X))
        if hasattr(self._model, "estimators_"):
            tree_matrix = np.column_stack([t.predict(X) for t in self._model.estimators_])
            lows = np.maximum(0.1, np.percentile(tree_matrix, 5, axis=1))
            highs = np.maximum(0.1, np.percentile(tree_matrix, 95, axis=1))
        else:
            lows = preds.copy()
            highs = preds.copy()
        return preds, lows, highs

    def default_runtime(self, topology_workers: int, topology_worker_cores: int,
                        topology_worker_mem_gb: int, profile: str) -> float:
        """Предсказывает время для дефолтной Spark-конфигурации на данной топологии."""
        row = {
            "topology_workers": topology_workers,
            "topology_worker_cores": topology_worker_cores,
            "topology_worker_mem_gb": topology_worker_mem_gb,
            "profile": profile,
            **_SPARK_DEFAULTS,
        }
        pred, _, _ = self.predict_one(row)
        return pred


# ── DNN service ───────────────────────────────────────────────────────────────

def _build_dnn_module(architecture: dict):
    """Reconstruct _DNN nn.Module from architecture dict (no BN, use_log1p)."""
    import torch
    import torch.nn as nn

    in_dim = architecture["in_dim"]
    hidden = architecture["hidden"]
    dropout = architecture.get("dropout", 0.0)

    layers: list = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, 1))

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    return _Net()


class DnnPredictorService:
    """Wraps a PyTorch DNN, exposes the same interface as PredictorService."""

    def __init__(self, artifact: ModelArtifact):
        self._pre = artifact.preprocessor
        self._job_type = artifact.job_type
        self._feature_cols: list[str] = list(self._pre.feature_names_in_)
        self._use_log1p: bool = artifact.architecture.get("use_log1p", True)
        self._np_layers: list[tuple[np.ndarray, np.ndarray]] | None = None
        weights_path = artifact.directory / "weights.npz" if artifact.directory is not None else None
        if weights_path is not None and weights_path.exists():
            with np.load(weights_path) as data:
                layer_count = int(data["layer_count"])
                self._np_layers = [
                    (data[f"layer_{idx}_weight"].astype(np.float32),
                     data[f"layer_{idx}_bias"].astype(np.float32))
                    for idx in range(layer_count)
                ]
            self._device = None
            self._net = None
            return

        import torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._net = _build_dnn_module(artifact.architecture).to(self._device)
        state = torch.load(artifact.model_path, map_location=self._device, weights_only=True)
        self._net.load_state_dict(state)
        self._net.eval()

    def _enrich(self, row: dict) -> dict:
        if "job_type" in self._feature_cols and "job_type" not in row:
            return {**row, "job_type": self._job_type}
        return row

    def _transform(self, rows: list[dict]) -> np.ndarray:
        enriched = [self._enrich(r) for r in rows]
        df = pd.DataFrame(enriched)[self._feature_cols]
        return self._pre.transform(df).astype(np.float32)

    def _net_predict(self, X: np.ndarray) -> np.ndarray:
        if self._np_layers is not None:
            y = X.astype(np.float32)
            for idx, (weight, bias) in enumerate(self._np_layers):
                y = y @ weight.T + bias
                if idx < len(self._np_layers) - 1:
                    y = np.maximum(y, 0)
            y_log = y.reshape(-1)
            if self._use_log1p:
                return np.clip(np.expm1(y_log), 0.1, None)
            return np.maximum(0.1, y_log)

        import torch
        with torch.no_grad():
            t = torch.from_numpy(X).to(self._device)
            y_log = self._net(t).cpu().numpy().reshape(-1)
        if self._use_log1p:
            return np.clip(np.expm1(y_log), 0.1, None)
        return np.maximum(0.1, y_log)

    def predict_one(self, row: dict) -> tuple[float, float, float]:
        X = self._transform([row])
        pred = float(self._net_predict(X)[0])
        return pred, pred, pred  # DNN has no confidence band

    def predict_batch(self, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = self._transform(rows)
        preds = self._net_predict(X)
        return preds, preds.copy(), preds.copy()

    def default_runtime(self, topology_workers: int, topology_worker_cores: int,
                        topology_worker_mem_gb: int, profile: str) -> float:
        row = {
            "topology_workers": topology_workers,
            "topology_worker_cores": topology_worker_cores,
            "topology_worker_mem_gb": topology_worker_mem_gb,
            "profile": profile,
            **_SPARK_DEFAULTS,
        }
        pred, _, _ = self.predict_one(row)
        return pred


# ── QL lookup service ─────────────────────────────────────────────────────────

class QlLookupService:
    """
    Returns pre-computed QL-optimal configs; uses RF PredictorService for time estimates.
    Falls back to RF's top-1 recommendation when the topology is not in the table.
    """

    def __init__(self, artifact: ModelArtifact, rf_svc: PredictorService):
        self._table: dict = artifact.best_cfg_table or {}
        self._rf = rf_svc
        self._job_type = artifact.job_type

    @staticmethod
    def _topo_key(workers: int, cores: int, mem_gb: int, profile: str) -> str:
        return f"{workers}w{cores}c{mem_gb}g_{profile}"

    def _nearest_key(self, workers: int, cores: int, profile: str) -> str | None:
        """Find the closest topology in the table by total-cores distance."""
        target_cores = workers * cores
        best_key, best_dist = None, float("inf")
        for key in self._table:
            try:
                parts = key.split("_")
                prof = parts[-1]
                if prof != profile:
                    continue
                topo = parts[0]
                w = int(topo.split("w")[0])
                c = int(topo.split("w")[1].split("c")[0])
                dist = abs(w * c - target_cores)
                if dist < best_dist:
                    best_dist, best_key = dist, key
            except (ValueError, IndexError):
                continue
        return best_key

    def lookup_cfg(self, workers: int, cores: int, mem_gb: int, profile: str) -> dict | None:
        key = self._topo_key(workers, cores, mem_gb, profile)
        if key in self._table:
            return self._table[key]
        nearest = self._nearest_key(workers, cores, profile)
        return self._table.get(nearest)

    def default_runtime(self, topology_workers: int, topology_worker_cores: int,
                        topology_worker_mem_gb: int, profile: str) -> float:
        return self._rf.default_runtime(
            topology_workers, topology_worker_cores, topology_worker_mem_gb, profile)
