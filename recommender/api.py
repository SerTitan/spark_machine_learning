import time
import base64
import binascii
import hashlib
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .auth import require_admin, verify_api_key
from .config_generator import config_distribution_warnings, generate_candidates
from .history import get_history, get_stats, init_db, log_request
from .inference import DnnPredictorService, PredictorService, QlLookupService
from .model_registry import ModelRegistry
from .schemas import (
    HealthResponse,
    AdminModelUploadRequest,
    MetricsResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    RecommendRequest,
    RecommendResponse,
    RecommendationItem,
    SparkConfigOut,
)
from .settings import settings

_SUPPORTED_JOB_TYPES = ["pagerank"]

registry = ModelRegistry(settings.model_dir)

# (job_type, model_name) → PredictorService | DnnPredictorService
_services: dict[tuple[str, str], PredictorService | DnnPredictorService] = {}
# job_type → QlLookupService
_ql_services: dict[str, QlLookupService] = {}
_model_load_warnings: list[str] = []

_STATIC_DIR = Path(__file__).parent / "static"


def _load_services_from_registry() -> None:
    _services.clear()
    _ql_services.clear()
    _model_load_warnings.clear()
    registry.load_all(_SUPPORTED_JOB_TYPES)

    for jt, mn in registry.loaded_models():
        artifact = registry.get(jt, mn)
        if artifact is None or mn == "ql":
            continue
        if mn == "dnn":
            try:
                _services[(jt, mn)] = DnnPredictorService(artifact)
            except ModuleNotFoundError as exc:
                if getattr(exc, "name", None) != "torch":
                    raise
                _model_load_warnings.append(
                    f"{jt}/{mn} skipped: PyTorch is not installed in this runtime"
                )
        else:
            _services[(jt, mn)] = PredictorService(artifact)

    for jt, mn in registry.loaded_models():
        if mn != "ql":
            continue
        artifact = registry.get(jt, mn)
        rf_svc = _services.get((jt, "rf"))
        if artifact is not None and rf_svc is not None:
            _ql_services[jt] = QlLookupService(artifact, rf_svc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _load_services_from_registry()
    yield


app = FastAPI(
    title="Spark Config Recommender",
    version="0.3.0",
    description=(
        "REST API для предсказания времени выполнения Spark-задач "
        "и рекомендации оптимальных конфигурационных параметров. "
        "Поддерживает модели: rf, dnn, ql."
    ),
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ── Вспомогательные функции ──────────────────────────────────────────────────

def _rank_candidates(
    candidates: list[dict],
    preds: "np.ndarray",
    lows: "np.ndarray",
    highs: "np.ndarray",
    default_time: float,
    top_k: int,
    profile: str,
) -> list[RecommendationItem]:
    import numpy as np
    out: list[RecommendationItem] = []
    seen: set[tuple] = set()
    profile_warn = [] if profile in ("small", "large") else [f"profile '{profile}' is outside training distribution"]

    for idx in preds.argsort():
        c = candidates[idx]
        key = (c["executor_cores"], c["executor_memory_mb"], c["executor_instances"])
        if key in seen:
            continue
        seen.add(key)
        speedup = round(default_time / float(preds[idx]), 3) if preds[idx] > 0 else 1.0
        out.append(RecommendationItem(
            rank=len(out) + 1,
            config=SparkConfigOut(**{k: bool(v) if k in ("shuffle_compress", "spill_compress",
                                     "broadcast_compress", "rdd_compress") else v
                                     for k, v in c.items()
                                     if k in SparkConfigOut.model_fields}),
            predicted_runtime_s=round(float(preds[idx]), 2),
            predicted_speedup_vs_default=speedup,
            confidence_band=[round(float(lows[idx]), 2), round(float(highs[idx]), 2)],
            warnings=profile_warn,
        ))
        if len(out) >= top_k:
            break
    return out


def _stable_seed(*parts) -> int:
    payload = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=True)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def _get_predictor_svc(job_type: str, model_name: str) -> PredictorService | DnnPredictorService:
    if job_type not in _SUPPORTED_JOB_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"job_type '{job_type}' is not supported. Supported: {_SUPPORTED_JOB_TYPES}",
        )
    svc = _services.get((job_type, model_name))
    if svc is None:
        available = [f"{jt}/{mn}" for jt, mn in _services]
        raise HTTPException(
            status_code=422,
            detail=f"Model '{job_type}/{model_name}' not loaded. Available: {available}",
        )
    return svc


def _resource_warnings(
    cfg: dict,
    topology_workers: int,
    topology_worker_cores: int,
    topology_worker_mem_gb: int,
) -> list[str]:
    """Check recommended config for potential resource over-commitment."""
    warns: list[str] = []
    executor_total_mb = cfg.get("executor_memory_mb", 0) * cfg.get("executor_instances", 1)
    cluster_mem_mb = topology_workers * topology_worker_mem_gb * 1024
    if executor_total_mb > cluster_mem_mb * 0.9:
        warns.append(
            f"Совокупная память executors ({executor_total_mb // 1024} ГБ) "
            f"близка к ресурсам кластера ({cluster_mem_mb // 1024} ГБ) — риск OOM"
        )
    total_cores_used = cfg.get("executor_cores", 1) * cfg.get("executor_instances", 1)
    total_cluster_cores = topology_workers * topology_worker_cores
    if total_cores_used > total_cluster_cores:
        warns.append(
            f"Суммарные ядра executors ({total_cores_used}) > "
            f"ядер кластера ({total_cluster_cores}) — возможна очередь YARN"
        )
    return warns


def _model_info(job_type: str, model_name: str) -> ModelInfo:
    artifact = registry.get(job_type, model_name)
    if artifact is None:
        return ModelInfo(job_type=job_type, predictor_type=model_name,
                         dataset_version="unknown")
    metrics = artifact.metrics
    if model_name == "ql" and not metrics:
        rf_artifact = registry.get(job_type, "rf")
        if rf_artifact is not None:
            metrics = rf_artifact.metrics
    mae = metrics.get("MAE") or metrics.get("mae_s")
    r2 = metrics.get("R2") or metrics.get("r2")
    return ModelInfo(
        job_type=job_type,
        predictor_type=artifact.predictor_type,
        predictor_mae_s=round(mae, 4) if mae is not None else None,
        predictor_r2=round(r2, 4) if r2 is not None else None,
        dataset_version=artifact.dataset_version,
    )


def _backup_dir() -> Path:
    path = settings.model_dir / ".disabled_models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _assert_inside_model_dir(path: Path) -> Path:
    resolved = path.resolve()
    base = settings.model_dir.resolve()
    if resolved != base and base not in resolved.parents:
        raise HTTPException(status_code=400, detail="Model path is outside model_dir")
    return resolved


def _decode_b64(name: str, value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except binascii.Error as exc:
        raise HTTPException(status_code=422, detail=f"{name} is not valid base64") from exc


def _write_uploaded_rf(req: AdminModelUploadRequest) -> dict:
    import joblib

    if req.job_type not in _SUPPORTED_JOB_TYPES:
        raise HTTPException(status_code=422, detail=f"Unsupported job_type: {req.job_type}")
    if req.model_name != "rf":
        raise HTTPException(status_code=422, detail="Only RF sklearn joblib upload is supported")

    jt_dir = _assert_inside_model_dir(settings.model_dir / req.job_type)
    target_dir = _assert_inside_model_dir(jt_dir / "rf")
    staging = _assert_inside_model_dir(settings.model_dir / ".uploads" / uuid.uuid4().hex)
    staging.mkdir(parents=True, exist_ok=False)

    try:
        (staging / "model.joblib").write_bytes(_decode_b64("content_b64", req.content_b64))

        if req.preprocessor_content_b64:
            (staging / "preprocessor.joblib").write_bytes(
                _decode_b64("preprocessor_content_b64", req.preprocessor_content_b64)
            )
        else:
            existing = registry.get(req.job_type, "rf")
            existing_pre = existing.directory / "preprocessor.joblib" if existing and existing.directory else None
            flat_pre = jt_dir / "preprocessor.joblib"
            pre_src = existing_pre if existing_pre and existing_pre.exists() else flat_pre
            if not pre_src.exists():
                raise HTTPException(
                    status_code=422,
                    detail="preprocessor_content_b64 is required when no current preprocessor exists",
                )
            shutil.copy2(pre_src, staging / "preprocessor.joblib")

        if req.report_content_b64:
            report = json.loads(_decode_b64("report_content_b64", req.report_content_b64).decode("utf-8"))
        else:
            report = {
                "dataset": "uploaded",
                "models": {"Uploaded_RF": {"metrics": {}, "best_params": None}},
            }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        model = joblib.load(staging / "model.joblib")
        preprocessor = joblib.load(staging / "preprocessor.joblib")
        if not hasattr(model, "predict"):
            raise HTTPException(status_code=422, detail="Uploaded model has no predict() method")
        if not hasattr(preprocessor, "transform") or not hasattr(preprocessor, "feature_names_in_"):
            raise HTTPException(status_code=422, detail="Uploaded preprocessor is incompatible")

        jt_dir.mkdir(parents=True, exist_ok=True)
        marker = jt_dir / ".disabled_rf"
        if marker.exists():
            marker.unlink()
        if target_dir.exists():
            backup = _backup_dir() / f"{req.job_type}_rf_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            shutil.move(str(target_dir), str(backup))
        shutil.move(str(staging), str(target_dir))
    except HTTPException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise HTTPException(status_code=422, detail=f"Uploaded model cannot be loaded: {exc}") from exc

    _load_services_from_registry()
    return {
        "status": "uploaded",
        "model": f"{req.job_type}/rf",
        "directory": str(target_dir),
    }


# ── Веб-интерфейс ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    return (_STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ── Service endpoints ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["service"])
async def health():
    loaded = [f"{jt}/{mn}" for jt, mn in _services] + [f"{jt}/ql" for jt in _ql_services]
    return HealthResponse(
        status="ok" if (_services or _ql_services) else "no_models_loaded",
        models_loaded=loaded,
        model_dir=str(settings.model_dir),
        model_warnings=list(_model_load_warnings),
    )


@app.get("/jobs", tags=["service"])
async def jobs():
    return {
        "supported": _SUPPORTED_JOB_TYPES,
        "models_loaded": [{"job_type": jt, "model": mn} for jt, mn in registry.loaded_models()],
    }


@app.get("/metrics", response_model=MetricsResponse, tags=["service"])
async def metrics():
    result = {}
    for jt, mn in registry.loaded_models():
        artifact = registry.get(jt, mn)
        metrics = artifact.metrics
        if mn == "ql" and not metrics:
            rf_artifact = registry.get(jt, "rf")
            if rf_artifact is not None:
                metrics = rf_artifact.metrics
        result[f"{jt}/{mn}"] = {
            "predictor_type": artifact.predictor_type,
            "metrics": metrics,
            "dataset_version": artifact.dataset_version,
        }
    return MetricsResponse(
        models=result,
        dataset_version="see per-model",
        model_dir=str(settings.model_dir),
    )


# ── History endpoints ─────────────────────────────────────────────────────────

@app.get("/history", tags=["history"])
async def history(limit: int = 50, offset: int = 0):
    return {"items": get_history(limit=limit, offset=offset)}


@app.get("/history/stats", tags=["history"])
async def history_stats():
    return get_stats()


# ── Prediction & Recommendation ───────────────────────────────────────────────

_422 = {422: {"description": "Model not loaded or invalid parameters"}}

ApiKey = Annotated[str | None, Depends(verify_api_key)]
AdminRole = Annotated[str, Depends(require_admin)]


@app.post("/predict", response_model=PredictResponse, tags=["prediction"], responses=_422)
async def predict(req: PredictRequest, key_hint: ApiKey):
    t0 = time.perf_counter()
    model_name = req.model_name
    svc_name = "rf" if model_name == "ql" else model_name
    svc = _get_predictor_svc(req.job_type, svc_name)
    row = req.model_dump(exclude={"model_name"})
    warnings: list[str] = []
    if model_name == "ql":
        warnings.append("RF+QL uses the RF predictor for runtime estimation; Q-learning is the optimizer")

    if req.executor_cores > req.topology_worker_cores:
        warnings.append(f"executor_cores ({req.executor_cores}) > worker_cores ({req.topology_worker_cores})")
    if req.executor_memory_mb > req.topology_worker_mem_gb * 1024:
        warnings.append(f"executor_memory_mb ({req.executor_memory_mb}) > worker RAM ({req.topology_worker_mem_gb * 1024} MB)")
    warnings.extend(config_distribution_warnings(
        row,
        req.topology_workers,
        req.topology_worker_cores,
        req.topology_worker_mem_gb,
    ))

    pred, low, high = svc.predict_one(row)
    result = PredictResponse(
        predicted_runtime_s=round(pred, 2),
        confidence_band=[round(low, 2), round(high, 2)],
        model_type=_model_info(req.job_type, model_name).predictor_type,
        warnings=warnings,
        model_info=_model_info(req.job_type, model_name),
    )

    log_request("/predict", req.job_type, req.model_dump(), result.model_dump(),
                (time.perf_counter() - t0) * 1000, key_hint)
    return result


@app.post("/recommend", response_model=RecommendResponse, tags=["recommendation"], responses=_422)
async def recommend(req: RecommendRequest, key_hint: ApiKey):
    t0 = time.perf_counter()
    model_name = req.model_name
    top_k = req.preferences.return_top_k if req.preferences else settings.default_top_k
    con = req.constraints.model_dump(exclude_none=True) if req.constraints else None

    # QL path: seed the search with the learned config, then score a
    # candidate pool with DNN so top_k can contain meaningful alternatives.
    if model_name == "ql":
        ql_svc = _ql_services.get(req.job_type)
        if ql_svc is None:
            raise HTTPException(status_code=422,
                                detail=f"QL model not loaded for job_type '{req.job_type}'")

        ql_cfg = ql_svc.lookup_cfg(
            req.topology.workers, req.topology.worker_cores,
            req.topology.worker_memory_gb, req.input.profile,
        )
        if ql_cfg is None:
            raise HTTPException(status_code=422,
                                detail="No pre-computed QL config for this topology")

        rf_svc = _get_predictor_svc(req.job_type, "rf")
        ql_row = {
            "topology_workers": req.topology.workers,
            "topology_worker_cores": req.topology.worker_cores,
            "topology_worker_mem_gb": req.topology.worker_memory_gb,
            "profile": req.input.profile,
            **ql_cfg,
        }
        candidates = [ql_row] + generate_candidates(
            topology_workers=req.topology.workers,
            topology_worker_cores=req.topology.worker_cores,
            topology_worker_mem_gb=req.topology.worker_memory_gb,
            profile=req.input.profile,
            n=max(settings.n_candidates, top_k * 20),
            constraints=con,
        )
        preds, lows, highs = rf_svc.predict_batch(candidates)
        default_time = rf_svc.default_runtime(
            req.topology.workers, req.topology.worker_cores,
            req.topology.worker_memory_gb, req.input.profile,
        )
        recommendations = _rank_candidates(
            candidates, preds, lows, highs,
            default_time, top_k, req.input.profile,
        )
        for rec in recommendations:
            cfg_dict = rec.config.model_dump()
            rec.warnings = list(rec.warnings) + _resource_warnings(
                cfg_dict, req.topology.workers,
                req.topology.worker_cores, req.topology.worker_memory_gb,
            )
        result = RecommendResponse(recommendations=recommendations,
                                   model_info=_model_info(req.job_type, "ql"))
        log_request("/recommend", req.job_type, req.model_dump(), result.model_dump(),
                    (time.perf_counter() - t0) * 1000, key_hint)
        return result

    # RF / DNN path: generate candidates, score, rank
    svc = _get_predictor_svc(req.job_type, model_name)

    candidates = generate_candidates(
        topology_workers=req.topology.workers,
        topology_worker_cores=req.topology.worker_cores,
        topology_worker_mem_gb=req.topology.worker_memory_gb,
        profile=req.input.profile,
        n=settings.n_candidates,
        constraints=con,
    )
    if not candidates:
        raise HTTPException(status_code=422, detail="No valid candidates after applying constraints")

    preds, lows, highs = svc.predict_batch(candidates)
    default_time = svc.default_runtime(
        req.topology.workers, req.topology.worker_cores,
        req.topology.worker_memory_gb, req.input.profile,
    )

    recommendations = _rank_candidates(candidates, preds, lows, highs,
                                        default_time, top_k, req.input.profile)

    # Append resource-safety warnings to each recommendation
    for rec in recommendations:
        cfg_dict = rec.config.model_dump()
        extra = _resource_warnings(cfg_dict, req.topology.workers,
                                   req.topology.worker_cores, req.topology.worker_memory_gb)
        rec.warnings = list(rec.warnings) + extra

    result = RecommendResponse(recommendations=recommendations,
                               model_info=_model_info(req.job_type, model_name))
    log_request("/recommend", req.job_type, req.model_dump(), result.model_dump(),
                (time.perf_counter() - t0) * 1000, key_hint)
    return result


# ── Admin endpoints ───────────────────────────────────────────────────────────

@app.get("/admin/models", tags=["admin"])
async def admin_list_models(role: AdminRole):
    result = {}
    for jt, mn in registry.loaded_models():
        artifact = registry.get(jt, mn)
        metrics = artifact.metrics
        if mn == "ql" and not metrics:
            rf_artifact = registry.get(jt, "rf")
            if rf_artifact is not None:
                metrics = rf_artifact.metrics
        result[f"{jt}/{mn}"] = {
            "predictor_type": artifact.predictor_type,
            "metrics": metrics,
            "dataset_version": artifact.dataset_version,
            "directory": str(artifact.directory),
        }
    return {"models": result, "model_dir": str(settings.model_dir), "role": role}


@app.post("/admin/models/reload", tags=["admin"])
async def admin_reload_models(role: AdminRole):
    _load_services_from_registry()
    return {
        "status": "reloaded",
        "models_loaded": [f"{jt}/{mn}" for jt, mn in _services] + [f"{jt}/ql" for jt in _ql_services],
        "role": role,
    }


@app.delete("/admin/models/{job_type}/{model_name}", tags=["admin"])
async def admin_delete_model(job_type: str, model_name: str, role: AdminRole):
    artifact = registry.get(job_type, model_name)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Model '{job_type}/{model_name}' not loaded")
    if artifact.directory is None:
        raise HTTPException(status_code=409, detail="Model has no artifact directory")

    source = _assert_inside_model_dir(artifact.directory)
    jt_dir = _assert_inside_model_dir(settings.model_dir / job_type)
    if not source.exists():
        raise HTTPException(status_code=404, detail="Artifact directory no longer exists")

    if source == jt_dir:
        destination = _backup_dir() / f"{job_type}_{model_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        shutil.move(str(source), str(destination))
    else:
        marker = jt_dir / f".disabled_{model_name}"
        marker.touch()
        destination = _backup_dir() / f"{job_type}_{model_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        shutil.move(str(source), str(destination))

    registry.remove(job_type, model_name)
    _services.pop((job_type, model_name), None)
    if model_name in ("rf", "ql"):
        _ql_services.pop(job_type, None)

    return {
        "status": "deleted",
        "model": f"{job_type}/{model_name}",
        "moved_to": str(destination),
        "role": role,
    }


@app.post("/admin/models/upload", tags=["admin"])
async def admin_upload_model(req: AdminModelUploadRequest, role: AdminRole):
    result = _write_uploaded_rf(req)
    result["role"] = role
    return result
