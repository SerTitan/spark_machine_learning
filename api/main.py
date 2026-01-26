"""
Мини-API для предсказания времени выполнения Spark.

Использование:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    POST /predict - предсказать время для конфигурации
    GET /health - проверка работоспособности
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# === Конфигурация ===
MODEL_DIR = Path(__file__).parent.parent / "out" / "final_best" / "baseline"
RF_MODEL_PATH = MODEL_DIR / "model_randomforest_simulatedannealing.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

# === Pydantic модели ===
class SparkConfig(BaseModel):
    """Конфигурация Spark для предсказания."""
    # Топология кластера
    topology_workers: int = Field(3, ge=1, le=12, description="Количество worker узлов")
    topology_worker_cores: int = Field(3, ge=1, le=16, description="Ядер на worker")
    topology_worker_mem_gb: int = Field(6, ge=1, le=64, description="Памяти на worker (GB)")

    # Профиль задачи
    profile: str = Field("large", description="Профиль: small, medium, large, huge")

    # Параметры Spark
    executor_cores: int = Field(2, ge=1, le=8, description="Ядер на executor")
    executor_memory_mb: int = Field(2048, ge=512, le=16384, description="Памяти на executor (MB)")
    executor_instances: int = Field(2, ge=1, le=12, description="Количество executors")
    driver_cores: int = Field(1, ge=1, le=4, description="Ядер на driver")
    driver_memory_mb: int = Field(1024, ge=512, le=8192, description="Памяти на driver (MB)")

    # Memory management
    memory_fraction: float = Field(0.6, ge=0.1, le=0.9, description="spark.memory.fraction")
    memory_storageFraction: float = Field(0.5, ge=0.1, le=0.9, description="spark.memory.storageFraction")

    # Shuffle & compression
    shuffle_compress: int = Field(1, ge=0, le=1, description="spark.shuffle.compress (0/1)")
    spill_compress: int = Field(1, ge=0, le=1, description="spark.shuffle.spill.compress (0/1)")
    shuffle_file_buffer_kb: int = Field(32, ge=16, le=256, description="spark.shuffle.file.buffer (KB)")

    # Broadcast
    broadcast_block_mb: int = Field(4, ge=2, le=32, description="spark.broadcast.blockSize (MB)")
    broadcast_compress: int = Field(1, ge=0, le=1, description="spark.broadcast.compress (0/1)")

    # Network
    maxSizeInFlight_mb: int = Field(48, ge=24, le=128, description="spark.reducer.maxSizeInFlight (MB)")
    rpc_message_maxSize: int = Field(128, ge=64, le=512, description="spark.rpc.message.maxSize (MB)")

    # Compression
    rdd_compress: int = Field(0, ge=0, le=1, description="spark.rdd.compress (0/1)")
    io_codec: str = Field("snappy", description="spark.io.compression.codec: lz4 или snappy")

    class Config:
        json_schema_extra = {
            "example": {
                "topology_workers": 3,
                "topology_worker_cores": 3,
                "topology_worker_mem_gb": 6,
                "profile": "large",
                "executor_cores": 3,
                "executor_memory_mb": 3072,
                "executor_instances": 3,
                "driver_cores": 1,
                "driver_memory_mb": 2048,
                "memory_fraction": 0.5,
                "memory_storageFraction": 0.4,
                "shuffle_compress": 0,
                "spill_compress": 0,
                "shuffle_file_buffer_kb": 128,
                "broadcast_block_mb": 20,
                "broadcast_compress": 0,
                "maxSizeInFlight_mb": 88,
                "rpc_message_maxSize": 160,
                "rdd_compress": 1,
                "io_codec": "lz4"
            }
        }


class PredictionResponse(BaseModel):
    """Ответ с предсказанием."""
    predicted_time_seconds: float = Field(..., description="Предсказанное время выполнения (секунды)")
    model_type: str = Field("RandomForest_SimulatedAnnealing", description="Тип использованной модели")
    warnings: list[str] = Field(default_factory=list, description="Предупреждения")


# === Загрузка модели ===
rf_model = None
preprocessor = None


def load_model():
    """Загружает модель при старте."""
    global rf_model, preprocessor

    if not RF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {RF_MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")

    rf_model = joblib.load(RF_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print(f"[API] Model loaded from {MODEL_DIR}")


# === FastAPI приложение ===
app = FastAPI(
    title="Spark Performance Predictor",
    description="API для предсказания времени выполнения Spark задач",
    version="0.1.0",
)


@app.on_event("startup")
async def startup_event():
    """Загружаем модель при старте."""
    try:
        load_model()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")


@app.get("/health")
async def health_check():
    """Проверка работоспособности."""
    return {
        "status": "ok" if rf_model is not None else "model_not_loaded",
        "model_loaded": rf_model is not None,
        "model_path": str(MODEL_DIR),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(config: SparkConfig):
    """
    Предсказывает время выполнения для заданной конфигурации Spark.

    Принимает конфигурацию кластера и параметры Spark,
    возвращает предсказанное время в секундах.
    """
    if rf_model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    warnings = []

    # Проверка физических ограничений
    if config.executor_cores > config.topology_worker_cores:
        warnings.append(
            f"executor_cores ({config.executor_cores}) > topology_worker_cores ({config.topology_worker_cores}). "
            "Spark ограничит executor_cores."
        )

    if config.executor_instances > config.topology_workers:
        warnings.append(
            f"executor_instances ({config.executor_instances}) > topology_workers ({config.topology_workers}). "
            "Не все executors смогут запуститься."
        )

    max_exec_mem = config.topology_worker_mem_gb * 1024 - 512
    if config.executor_memory_mb > max_exec_mem:
        warnings.append(
            f"executor_memory_mb ({config.executor_memory_mb}) > доступно на worker ({max_exec_mem}MB). "
            "Возможна нехватка памяти."
        )

    # Формируем DataFrame для предсказания
    data = {
        "topology_workers": config.topology_workers,
        "topology_worker_cores": config.topology_worker_cores,
        "topology_worker_mem_gb": config.topology_worker_mem_gb,
        "profile": config.profile,
        "executor_cores": config.executor_cores,
        "executor_memory_mb": config.executor_memory_mb,
        "executor_instances": config.executor_instances,
        "driver_cores": config.driver_cores,
        "driver_memory_mb": config.driver_memory_mb,
        "memory_fraction": config.memory_fraction,
        "memory_storageFraction": config.memory_storageFraction,
        "shuffle_compress": config.shuffle_compress,
        "spill_compress": config.spill_compress,
        "shuffle_file_buffer_kb": config.shuffle_file_buffer_kb,
        "broadcast_block_mb": config.broadcast_block_mb,
        "broadcast_compress": config.broadcast_compress,
        "maxSizeInFlight_mb": config.maxSizeInFlight_mb,
        "rpc_message_maxSize": config.rpc_message_maxSize,
        "rdd_compress": config.rdd_compress,
        "io_codec": config.io_codec,
    }

    df = pd.DataFrame([data])

    try:
        # Препроцессинг и предсказание
        X_transformed = preprocessor.transform(df)
        prediction = rf_model.predict(X_transformed)[0]

        # Убедимся что время положительное
        prediction = max(0.1, float(prediction))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictionResponse(
        predicted_time_seconds=round(prediction, 2),
        model_type="RandomForest_SimulatedAnnealing",
        warnings=warnings,
    )


# === Запуск для отладки ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
