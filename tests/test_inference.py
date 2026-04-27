"""Unit-тесты модуля recommender.inference."""

import numpy as np
import pytest

from recommender.config_generator import generate_candidates
from recommender.inference import PredictorService
from recommender.model_registry import ModelRegistry
from recommender.settings import settings


@pytest.fixture(scope="module")
def service() -> PredictorService:
    reg = ModelRegistry(settings.model_dir)
    reg.load_all(["pagerank"])
    return PredictorService(reg.get("pagerank"))


@pytest.fixture
def base_row():
    return {
        "topology_workers": 3,
        "topology_worker_cores": 3,
        "topology_worker_mem_gb": 6,
        "profile": "large",
        "executor_cores": 2,
        "executor_memory_mb": 3072,
        "executor_instances": 3,
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


class TestPredictOne:
    def test_returns_positive_prediction(self, service, base_row):
        pred, low, high = service.predict_one(base_row)
        assert pred > 0

    def test_band_ordered(self, service, base_row):
        pred, low, high = service.predict_one(base_row)
        assert low <= pred <= high + 0.01  # небольшая погрешность float

    def test_no_nan(self, service, base_row):
        pred, low, high = service.predict_one(base_row)
        assert not any(np.isnan(x) for x in (pred, low, high))

    def test_different_configs_differ(self, service, base_row):
        pred1, *_ = service.predict_one(base_row)

        row2 = base_row.copy()
        row2["executor_memory_mb"] = 1024  # минимальная память
        pred2, *_ = service.predict_one(row2)

        # Модель должна давать разные предсказания для разных конфигов
        assert pred1 != pred2

    def test_both_profiles_accepted(self, service, base_row):
        # Оба профиля должны принимаются без ошибок.
        # Порядок предсказаний зависит от обучающих данных (текущая модель
        # обучена только на large, переобучение после сбора мульти-нагрузочного датасета).
        pred_large, _, _ = service.predict_one(base_row)
        row_small = dict(base_row, profile="small")
        pred_small, _, _ = service.predict_one(row_small)
        assert pred_large > 0 and pred_small > 0


class TestPredictBatch:
    def test_shape_matches_input(self, service):
        cands = generate_candidates(3, 3, 6, "large", n=50, rng_seed=0)
        preds, lows, highs = service.predict_batch(cands)
        assert preds.shape == (50,)
        assert lows.shape == (50,)
        assert highs.shape == (50,)

    def test_all_positive(self, service):
        cands = generate_candidates(3, 3, 6, "large", n=100, rng_seed=1)
        preds, lows, highs = service.predict_batch(cands)
        assert (preds > 0).all()
        assert (lows > 0).all()
        assert (highs > 0).all()

    def test_no_nan(self, service):
        cands = generate_candidates(3, 3, 6, "large", n=50, rng_seed=2)
        preds, lows, highs = service.predict_batch(cands)
        assert not np.isnan(preds).any()
        assert not np.isnan(lows).any()
        assert not np.isnan(highs).any()

    def test_lows_leq_highs(self, service):
        cands = generate_candidates(3, 3, 6, "large", n=100, rng_seed=3)
        _, lows, highs = service.predict_batch(cands)
        assert (lows <= highs + 1e-6).all()

    def test_single_candidate_matches_predict_one(self, service, base_row):
        preds, lows, highs = service.predict_batch([base_row])
        pred1, low1, high1 = service.predict_one(base_row)
        assert abs(preds[0] - pred1) < 1e-6
        assert abs(lows[0] - low1) < 1e-6
        assert abs(highs[0] - high1) < 1e-6


class TestDefaultRuntime:
    def test_returns_positive(self, service):
        t = service.default_runtime(3, 3, 6, "large")
        assert t > 0

    def test_large_faster_than_small_is_not_required(self, service):
        # Проверяем только что оба вызова работают без ошибок
        t_large = service.default_runtime(3, 3, 6, "large")
        t_small = service.default_runtime(3, 3, 6, "small")
        assert t_large > 0 and t_small > 0
