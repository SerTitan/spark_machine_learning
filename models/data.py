"""
Модуль загрузки и препроцессинга данных для оптимизации Spark.

Формат CSV (из collect_wordcount_data.sh):
- topology_workers, topology_worker_cores, topology_worker_mem_gb
- profile
- executor_cores, executor_memory, executor_instances
- driver_cores, driver_memory
- memory_fraction, memory_storageFraction
- shuffle_compress, spill_compress, shuffle_file_buffer
- broadcast_block, broadcast_compress
- maxSizeInFlight, io_codec, rpc_message_maxSize, rdd_compress
- median_duration_s, exit_code
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# === Определение фич ===
# Эти списки определяют какие колонки из CSV относятся к каким типам данных

# Топология кластера - описывает железо на котором запускается Spark
TOPOLOGY_FEATURES = [
    "topology_workers",
    "topology_worker_cores",
    "topology_worker_mem_gb",
]

# Параметры Spark которые мы оптимизируем (16 штук)
SPARK_PARAMETERS = [
    "executor_cores",           # Кол-во ядер на executor
    "executor_memory",          # Память executor (1g, 2g и т.д.)
    "executor_instances",       # Кол-во executor'ов
    "driver_cores",             # Кол-во ядер на driver
    "driver_memory",            # Память driver
    "memory_fraction",          # Доля памяти для вычислений
    "memory_storageFraction",   # Доля памяти для кеширования
    "shuffle_compress",         # Сжимать ли shuffle данные
    "spill_compress",           # Сжимать ли spill данные
    "shuffle_file_buffer",      # Размер буфера shuffle
    "broadcast_block",          # Размер блока broadcast
    "broadcast_compress",       # Сжимать ли broadcast
    "maxSizeInFlight",          # Макс. размер данных в полёте
    "io_codec",                 # Кодек сжатия (lz4, snappy, zstd)
    "rpc_message_maxSize",      # Макс. размер RPC сообщения
    "rdd_compress",             # Сжимать ли RDD
]

# Категориальные фичи - их нужно one-hot кодировать
CATEGORICAL_FEATURES = ["profile", "io_codec"]

# Булевы фичи - true/false конвертируем в 1/0
BOOLEAN_FEATURES = [
    "shuffle_compress",
    "spill_compress",
    "broadcast_compress",
    "rdd_compress",
]

# Фичи с памятью - строки вида "1g", "512m" конвертируем в числа (MB)
MEMORY_FEATURES = [
    "executor_memory",
    "driver_memory",
    "shuffle_file_buffer",
    "broadcast_block",
    "maxSizeInFlight",
]

NUMERIC_FEATURES = [
    "topology_workers",
    "topology_worker_cores",
    "topology_worker_mem_gb",
    "executor_cores",
    "executor_instances",
    "driver_cores",
    "memory_fraction",
    "memory_storageFraction",
    "rpc_message_maxSize",
    # После конвертации памяти в MB:
    "executor_memory_mb",
    "driver_memory_mb",
    "shuffle_file_buffer_kb",
    "broadcast_block_mb",
    "maxSizeInFlight_mb",
]

TARGET_COLUMN = "median_duration_s"


def parse_memory_to_mb(val: Any) -> float:
    """Конвертирует строку памяти (1g, 512m, 32k) в MB."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return np.nan
    s = str(val).strip().lower()
    # Парсим суффикс и конвертируем в мегабайты
    if s.endswith("g"):
        return float(s[:-1]) * 1024.0  # 1g = 1024 MB
    if s.endswith("m"):
        return float(s[:-1])            # уже в MB
    if s.endswith("k"):
        return float(s[:-1]) / 1024.0   # KB -> MB
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_memory_to_kb(val: Any) -> float:
    """Конвертирует строку памяти в KB."""
    mb = parse_memory_to_mb(val)
    return mb * 1024.0 if not np.isnan(mb) else np.nan


def parse_bool(val: Any) -> int:
    """Конвертирует true/false в 1/0."""
    s = str(val).strip().lower()
    if s in ("true", "1", "yes"):
        return 1
    if s in ("false", "0", "no"):
        return 0
    try:
        return 1 if int(s) != 0 else 0
    except (ValueError, TypeError):
        return 0


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """
    Загружает CSV и выполняет базовую очистку.

    Args:
        csv_path: путь к CSV файлу

    Returns:
        DataFrame с очищенными данными
    """
    # Читаем CSV файл собранный скриптом collect_wordcount_data.sh
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]  # убираем пробелы в названиях колонок

    # Оставляем только успешные запуски (exit_code = 0)
    if "exit_code" in df.columns:
        df = df[df["exit_code"] == 0].copy()

    # Проверяем что есть колонка с временем выполнения
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in CSV")

    # Удаляем строки где время выполнения некорректное
    df = df[pd.to_numeric(df[TARGET_COLUMN], errors="coerce") > 0].copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(float)

    return df


@dataclass
class SparkDataset:
    """
    Датасет для обучения моделей оптимизации Spark.

    Attributes:
        X_train, X_val, X_test: фичи (DataFrame)
        y_train, y_val, y_test: таргеты (Series)
        preprocessor: sklearn ColumnTransformer
        feature_names: список имён фич после препроцессинга
        numeric_cols: числовые колонки
        categorical_cols: категориальные колонки
        boolean_cols: булевы колонки
    """

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    feature_names: List[str]
    numeric_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    boolean_cols: List[str] = field(default_factory=list)
    raw_df: pd.DataFrame = field(default=None)

    @property
    def n_features(self) -> int:
        """Количество фич после препроцессинга."""
        return len(self.feature_names)

    @property
    def X_train_transformed(self) -> np.ndarray:
        """Трансформированные тренировочные фичи."""
        return self.preprocessor.transform(self.X_train)

    @property
    def X_val_transformed(self) -> np.ndarray:
        """Трансформированные валидационные фичи."""
        return self.preprocessor.transform(self.X_val)

    @property
    def X_test_transformed(self) -> np.ndarray:
        """Трансформированные тестовые фичи."""
        return self.preprocessor.transform(self.X_test)

    def get_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray, np.ndarray]:
        """Возвращает (X_tr, X_val, X_te, y_tr, y_val, y_te) в numpy."""
        return (
            self.X_train_transformed,
            self.X_val_transformed,
            self.X_test_transformed,
            self.y_train.values,
            self.y_val.values,
            self.y_test.values,
        )

    def get_param_grid(self) -> Dict[str, List[Any]]:
        """
        Возвращает пространство параметров для RL-оптимизации.
        Значения берутся из уникальных значений в датасете.
        """
        all_X = pd.concat([self.X_train, self.X_val, self.X_test])
        grid = {}

        # Числовые параметры Spark (не топология)
        spark_numeric = [
            "executor_cores", "executor_instances", "driver_cores",
            "memory_fraction", "memory_storageFraction", "rpc_message_maxSize",
            "executor_memory_mb", "driver_memory_mb",
            "shuffle_file_buffer_kb", "broadcast_block_mb", "maxSizeInFlight_mb",
        ]
        for col in spark_numeric:
            if col in all_X.columns:
                vals = sorted(all_X[col].dropna().unique().tolist())
                if vals:
                    grid[col] = vals

        # Булевы параметры
        for col in self.boolean_cols:
            if col in all_X.columns:
                grid[col] = [0, 1]

        # Категориальные (только io_codec, без profile)
        if "io_codec" in all_X.columns:
            grid["io_codec"] = sorted(all_X["io_codec"].unique().tolist())

        return grid


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Подготавливает фичи из сырого DataFrame.

    - Конвертирует память в числа (MB/KB)
    - Конвертирует булевы значения в 0/1
    """
    work = df.copy()

    # Конвертируем строки памяти ("1g", "512m") в числа для модели
    work["executor_memory_mb"] = work["executor_memory"].apply(parse_memory_to_mb)
    work["driver_memory_mb"] = work["driver_memory"].apply(parse_memory_to_mb)
    work["broadcast_block_mb"] = work["broadcast_block"].apply(parse_memory_to_mb)
    work["maxSizeInFlight_mb"] = work["maxSizeInFlight"].apply(parse_memory_to_mb)
    work["shuffle_file_buffer_kb"] = work["shuffle_file_buffer"].apply(parse_memory_to_kb)

    # Конвертируем "true"/"false" в 1/0
    for col in BOOLEAN_FEATURES:
        if col in work.columns:
            work[col] = work[col].apply(parse_bool)

    return work


def build_preprocessor(
    X: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    boolean_cols: List[str],
) -> Tuple[ColumnTransformer, List[str]]:
    """
    Строит sklearn ColumnTransformer для препроцессинга.

    - Числовые: StandardScaler (нормализация)
    - Категориальные: OneHotEncoder (one-hot кодирование)
    - Булевы: passthrough (оставляем как есть)

    Returns:
        (preprocessor, feature_names)
    """
    transformers = []
    feature_names = []

    # Числовые фичи - нормализуем (среднее=0, std=1)
    if numeric_cols:
        num_transformer = Pipeline([("scaler", StandardScaler())])
        transformers.append(("num", num_transformer, numeric_cols))
        feature_names.extend(numeric_cols)

    # Категориальные фичи - one-hot кодирование (profile, io_codec -> много колонок)
    if categorical_cols:
        try:
            cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            # Старая версия sklearn
            cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", cat_transformer, categorical_cols))

    # Булевы фичи - уже 0/1, оставляем без изменений
    if boolean_cols:
        transformers.append(("bool", "passthrough", boolean_cols))
        feature_names.extend(boolean_cols)

    # ColumnTransformer применяет трансформации к разным типам колонок
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    preprocessor.fit(X)

    # Собираем имена фич после one-hot кодирования (profile_small, profile_large и т.д.)
    if categorical_cols:
        cat_encoder = preprocessor.named_transformers_["cat"]
        if hasattr(cat_encoder, "get_feature_names_out"):
            cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        else:
            cat_names = []
            for i, col in enumerate(categorical_cols):
                cats = cat_encoder.categories_[i]
                cat_names.extend([f"{col}_{c}" for c in cats])
        # Вставляем имена категориальных фич после числовых
        idx = len(numeric_cols) if numeric_cols else 0
        feature_names = feature_names[:idx] + cat_names + feature_names[idx:]

    return preprocessor, feature_names


def create_dataset(
    csv_path: str | Path,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    target_log_transform: bool = False,
) -> SparkDataset:
    """
    Создаёт SparkDataset из CSV файла.

    Args:
        csv_path: путь к CSV
        test_size: доля тестовой выборки
        val_size: доля валидационной выборки (от train)
        random_state: seed для воспроизводимости
        target_log_transform: применить log1p к таргету

    Returns:
        SparkDataset с подготовленными данными
    """
    # Загрузка
    df = load_dataset(csv_path)

    # Подготовка фич
    df = prepare_features(df)

    # Определяем колонки
    numeric_cols = [
        "topology_workers",
        "topology_worker_cores",
        "topology_worker_mem_gb",
        "executor_cores",
        "executor_instances",
        "driver_cores",
        "memory_fraction",
        "memory_storageFraction",
        "rpc_message_maxSize",
        "executor_memory_mb",
        "driver_memory_mb",
        "shuffle_file_buffer_kb",
        "broadcast_block_mb",
        "maxSizeInFlight_mb",
    ]
    categorical_cols = ["profile", "io_codec"]
    boolean_cols = ["shuffle_compress", "spill_compress", "broadcast_compress", "rdd_compress"]

    # Оставляем только колонки которые есть в наших данных
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    boolean_cols = [c for c in boolean_cols if c in df.columns]

    all_feature_cols = numeric_cols + categorical_cols + boolean_cols

    # X - фичи (параметры Spark), y - таргет (время выполнения)
    X = df[all_feature_cols].copy()
    y = df[TARGET_COLUMN].copy()

    # log1p трансформация стабилизирует обучение при больших значениях времени
    if target_log_transform:
        y = np.log1p(y)

    # Удаляем строки с пропущенными значениями
    mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X[mask]
    y = y[mask]

    # Стратификация - одинаковое соотношение profile в train/test/val
    stratify = X["profile"] if "profile" in X.columns and len(X["profile"].unique()) > 1 else None

    # Разбиваем на train (64%) и test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Разбиваем train на train (80% от 80%) и val (20% от 80% = 16%)
    stratify_val = X_train["profile"] if stratify is not None else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=stratify_val
    )

    # Строим препроцессор ТОЛЬКО на train чтобы не было утечки данных
    preprocessor, feature_names = build_preprocessor(
        X_train, numeric_cols, categorical_cols, boolean_cols
    )

    return SparkDataset(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        boolean_cols=boolean_cols,
        raw_df=df,
    )


def get_default_config(dataset: SparkDataset, exclude_topology: bool = True) -> Dict[str, Any]:
    """
    Возвращает конфигурацию по умолчанию (медианы числовых, моды категориальных).

    Args:
        dataset: SparkDataset
        exclude_topology: исключить topology фичи (topology_workers и т.д.)
    """
    all_X = pd.concat([dataset.X_train, dataset.X_val])
    config = {}

    # Фичи топологии - исключаем если нужно
    topology_cols = {"topology_workers", "topology_worker_cores", "topology_worker_mem_gb"}

    for col in dataset.numeric_cols:
        if col in all_X.columns:
            if exclude_topology and col in topology_cols:
                continue
            config[col] = float(all_X[col].median())

    for col in dataset.boolean_cols:
        if col in all_X.columns:
            config[col] = int(all_X[col].mode().iloc[0])

    for col in dataset.categorical_cols:
        if col in all_X.columns and col != "profile":
            config[col] = str(all_X[col].mode().iloc[0])

    return config
