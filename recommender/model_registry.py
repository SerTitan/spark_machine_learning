"""
Загружает артефакты моделей с диска.

Структура директории (новая, multi-model):
    <model_dir>/<job_type>/rf/
        model.joblib, preprocessor.joblib, best_cfg.json, report.json
    <model_dir>/<job_type>/dnn/
        model.pt, architecture.json, preprocessor.joblib, best_cfg.json, report.json
    <model_dir>/<job_type>/ql/
        best_cfg_table.json, report.json

Обратная совместимость: если подпапок rf/dnn/ql нет, используется
старый формат (model_*.joblib) как "rf".
"""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib

_REPORT_JSON = "report.json"
_PREPROCESSOR = "preprocessor.joblib"
_DISABLED_PREFIX = ".disabled_"
_LEGACY_MODEL_NAMES = (
    "model_randomforest_randomsearch.joblib",
    "model_randomforest_simulatedannealing.joblib",
    "model_extratrees.joblib",
)


@dataclass
class ModelArtifact:
    job_type: str
    model_name: str           # "rf" | "dnn" | "ql"
    predictor: object         # sklearn model, or None for DNN/QL
    preprocessor: object
    predictor_type: str
    metrics: dict
    dataset_version: str
    directory: Path | None = None
    model_path: Path | None = None    # DNN: path to .pt file
    architecture: dict | None = None  # DNN: {in_dim, hidden, dropout, use_log1p}
    best_cfg_table: dict | None = None  # QL: {topo_key: config_dict}


def _read_report(directory: Path) -> tuple[dict, str, dict]:
    """Returns (metrics, dataset_version, report_models)."""
    report_path = directory / _REPORT_JSON
    if not report_path.exists():
        return {}, "unknown", {}
    with open(report_path) as f:
        report = json.load(f)
    dataset_version = report.get("dataset", report.get("dataset_version", "unknown"))
    report_models = report.get("models", {})
    metrics: dict = {}
    for info in report_models.values():
        metrics = info.get("metrics", {})
        break
    return metrics, dataset_version, report_models


def _find_rf_model_path(directory: Path, report_models: dict) -> tuple[Path, str, dict]:
    """Locate the sklearn model file; return (path, predictor_type, metrics).

    Priority: lowest-MAE from report → named model.joblib → legacy filenames.
    This mirrors the original code so that multi-model reports (containing
    baselines like Dummy_mean) correctly resolve to the best model.
    """
    # Lowest-MAE from report (original logic — works for both multi-model and
    # single-model reports, as long as the model file actually exists on disk)
    if report_models:
        candidates = []
        for mname, info in report_models.items():
            mp = directory / f"model_{mname.lower().replace(' ', '_')}.joblib"
            mae = info.get("metrics", {}).get("MAE")
            if mp.exists() and mae is not None:
                candidates.append((float(mae), mname, mp, info.get("metrics", {})))
        if candidates:
            _, mname, mp, metrics = min(candidates, key=lambda x: x[0])
            return mp, mname.lower().replace(" ", "_"), metrics

    # Named model.joblib — used by artifacts saved by train_compare_models.py
    named = directory / "model.joblib"
    if named.exists():
        metrics = {}
        predictor_type = "rf"
        if len(report_models) == 1:
            mname, info = next(iter(report_models.items()))
            predictor_type = mname.lower().replace(" ", "_")
            metrics = info.get("metrics", {})
        return named, predictor_type, metrics

    # Legacy filenames (old training pipeline)
    for legacy in _LEGACY_MODEL_NAMES:
        p = directory / legacy
        if p.exists():
            model_key = legacy.replace("model_", "").replace(".joblib", "")
            for mname, info in report_models.items():
                if mname.lower().replace(" ", "_") == model_key:
                    return p, model_key, info.get("metrics", {})
            return p, model_key, {}

    raise FileNotFoundError(f"No supported model file in {directory}")


def _load_rf_artifact(directory: Path, job_type: str) -> ModelArtifact:
    preprocessor = joblib.load(directory / _PREPROCESSOR)
    _, dataset_version, report_models = _read_report(directory)
    model_path, predictor_type, metrics = _find_rf_model_path(directory, report_models)
    predictor = joblib.load(model_path)
    return ModelArtifact(
        job_type=job_type, model_name="rf",
        predictor=predictor, preprocessor=preprocessor,
        predictor_type=predictor_type, metrics=metrics,
        dataset_version=dataset_version, directory=directory,
        model_path=model_path,
    )


def _load_dnn_artifact(directory: Path, job_type: str) -> ModelArtifact:
    preprocessor = joblib.load(directory / _PREPROCESSOR)
    metrics, dataset_version, _ = _read_report(directory)
    with open(directory / "architecture.json") as f:
        architecture = json.load(f)
    return ModelArtifact(
        job_type=job_type, model_name="dnn",
        predictor=None, preprocessor=preprocessor,
        predictor_type="dnn", metrics=metrics,
        dataset_version=dataset_version, directory=directory,
        model_path=directory / "model.pt",
        architecture=architecture,
    )


def _load_ql_artifact(directory: Path, job_type: str) -> ModelArtifact:
    _, dataset_version, _ = _read_report(directory)
    with open(directory / "best_cfg_table.json") as f:
        best_cfg_table = json.load(f)
    return ModelArtifact(
        job_type=job_type, model_name="ql",
        predictor=None, preprocessor=None,
        predictor_type="ql", metrics={},
        dataset_version=dataset_version, directory=directory,
        best_cfg_table=best_cfg_table,
    )


def _try_load(loader, directory: Path, job_type: str) -> ModelArtifact | None:
    try:
        return loader(directory, job_type)
    except Exception:
        return None


class ModelRegistry:
    def __init__(self, base_dir: Path):
        self._base = base_dir
        self._cache: dict[tuple[str, str], ModelArtifact] = {}

    def _load_named_models(self, jt: str, jt_dir: Path) -> bool:
        """Load rf/dnn/ql subdirs. Returns True if at least one loaded."""
        loaded = False
        for sub, loader, required_file in (
            ("rf",  _load_rf_artifact,  _PREPROCESSOR),
            ("dnn", _load_dnn_artifact, "model.pt"),
            ("ql",  _load_ql_artifact,  "best_cfg_table.json"),
        ):
            if (jt_dir / f"{_DISABLED_PREFIX}{sub}").exists():
                continue
            sub_dir = jt_dir / sub
            if sub_dir.is_dir() and (sub_dir / required_file).exists():
                artifact = _try_load(loader, sub_dir, jt)
                if artifact is not None:
                    self._cache[(jt, sub)] = artifact
                    loaded = True
        return loaded

    def load_all(self, supported_job_types: list[str]) -> None:
        self._cache.clear()
        for jt in supported_job_types:
            jt_dir = self._base / jt
            if not jt_dir.is_dir():
                continue
            if not self._load_named_models(jt, jt_dir):
                # Backward compat: old flat directory format
                if (jt_dir / f"{_DISABLED_PREFIX}rf").exists():
                    continue
                artifact = _try_load(_load_rf_artifact, jt_dir, jt)
                if artifact is not None:
                    artifact.model_name = "rf"
                    self._cache[(jt, "rf")] = artifact

    def get(self, job_type: str, model_name: str = "rf") -> ModelArtifact | None:
        return self._cache.get((job_type, model_name))

    def loaded_job_types(self) -> list[str]:
        return list({jt for jt, _ in self._cache})

    def loaded_models(self) -> list[tuple[str, str]]:
        return list(self._cache.keys())

    def remove(self, job_type: str, model_name: str) -> None:
        self._cache.pop((job_type, model_name), None)
