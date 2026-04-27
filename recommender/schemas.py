from typing import Optional
from pydantic import BaseModel, Field


# ── Входные схемы ────────────────────────────────────────────────────────────

class TopologySpec(BaseModel):
    workers: int = Field(..., ge=1, le=16)
    worker_cores: int = Field(..., ge=1, le=32)
    worker_memory_gb: int = Field(..., ge=1, le=256)


class InputSpec(BaseModel):
    profile: str = Field(..., description="Профиль HiBench: small | large")
    size_bytes: Optional[int] = Field(None, ge=0)


class ConstraintsSpec(BaseModel):
    max_executor_cores: Optional[int] = Field(None, ge=1)
    max_executor_memory_mb: Optional[int] = Field(None, ge=256)
    max_executor_instances: Optional[int] = Field(None, ge=1)


class PreferencesSpec(BaseModel):
    optimize_for: str = Field("runtime", description="Метрика оптимизации: runtime")
    return_top_k: int = Field(3, ge=1, le=10)


class RecommendRequest(BaseModel):
    job_type: str = Field(..., description="Тип нагрузки: сейчас поддерживается pagerank")
    input: InputSpec
    topology: TopologySpec
    constraints: Optional[ConstraintsSpec] = None
    preferences: Optional[PreferencesSpec] = None
    model_name: str = Field("rf", description="Стратегия рекомендации: rf | dnn | ql")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_type": "pagerank",
                "input": {"profile": "large"},
                "topology": {"workers": 3, "worker_cores": 3, "worker_memory_gb": 6},
                "constraints": {"max_executor_cores": 3, "max_executor_memory_mb": 4096},
                "preferences": {"return_top_k": 3},
                "model_name": "rf",
            }
        }
    }


class PredictRequest(BaseModel):
    job_type: str = Field("pagerank")
    profile: str = Field("large")
    model_name: str = Field("rf", description="Стратегия оценки: rf | dnn | ql")
    topology_workers: int = Field(..., ge=1)
    topology_worker_cores: int = Field(..., ge=1)
    topology_worker_mem_gb: int = Field(..., ge=1)
    executor_cores: int = Field(..., ge=1)
    executor_memory_mb: int = Field(..., ge=256)
    executor_instances: int = Field(..., ge=1)
    driver_cores: int = Field(1, ge=1)
    driver_memory_mb: int = Field(1024, ge=256)
    memory_fraction: float = Field(0.6, ge=0.1, le=0.9)
    memory_storageFraction: float = Field(0.5, ge=0.1, le=0.9)
    shuffle_compress: int = Field(1, ge=0, le=1)
    spill_compress: int = Field(1, ge=0, le=1)
    shuffle_file_buffer_kb: int = Field(32, ge=16)
    broadcast_block_mb: int = Field(4, ge=2)
    broadcast_compress: int = Field(1, ge=0, le=1)
    maxSizeInFlight_mb: int = Field(48, ge=24)
    rpc_message_maxSize: int = Field(128, ge=64)
    rdd_compress: int = Field(0, ge=0, le=1)
    io_codec: str = Field("lz4")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_type": "pagerank",
                "profile": "large",
                "topology_workers": 3,
                "topology_worker_cores": 3,
                "topology_worker_mem_gb": 6,
                "executor_cores": 2,
                "executor_memory_mb": 3072,
                "executor_instances": 3,
            }
        }
    }


class AdminModelUploadRequest(BaseModel):
    job_type: str = Field("pagerank")
    model_name: str = Field("rf", description="Сейчас поддерживается загрузка sklearn RF как rf")
    filename: str = Field(..., min_length=1)
    content_b64: str = Field(..., min_length=1)
    preprocessor_filename: Optional[str] = None
    preprocessor_content_b64: Optional[str] = None
    report_filename: Optional[str] = None
    report_content_b64: Optional[str] = None


# ── Выходные схемы ───────────────────────────────────────────────────────────

class SparkConfigOut(BaseModel):
    executor_cores: int
    executor_memory_mb: int
    executor_instances: int
    driver_cores: int
    driver_memory_mb: int
    memory_fraction: float
    memory_storageFraction: float
    shuffle_compress: bool
    spill_compress: bool
    shuffle_file_buffer_kb: int
    broadcast_block_mb: int
    broadcast_compress: bool
    maxSizeInFlight_mb: int
    rpc_message_maxSize: int
    rdd_compress: bool
    io_codec: str


class RecommendationItem(BaseModel):
    rank: int
    config: SparkConfigOut
    predicted_runtime_s: float
    predicted_speedup_vs_default: float
    confidence_band: list[float] = Field(description="[p5, p95] по деревьям RF")
    warnings: list[str]


class ModelInfo(BaseModel):
    job_type: str
    predictor_type: str
    predictor_mae_s: Optional[float] = None
    predictor_r2: Optional[float] = None
    dataset_version: str


class RecommendResponse(BaseModel):
    recommendations: list[RecommendationItem]
    model_info: ModelInfo


class PredictResponse(BaseModel):
    predicted_runtime_s: float
    confidence_band: list[float]
    model_type: str
    warnings: list[str]
    model_info: Optional[ModelInfo] = None


class MetricsResponse(BaseModel):
    models: dict
    dataset_version: str
    model_dir: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    model_dir: str
    model_warnings: list[str] = Field(default_factory=list)
