# Описание тестового покрытия

Последнее обновление: 2026-04-27.  
Запуск: `.venv/bin/pytest -v` (79 тестов, 0 failed, покрытие ≈ 93%).

---

## Структура тестов

```
tests/
├── conftest.py              # Фикстуры: client, recommend_body, predict_body
├── test_config_generator.py # Unit: генерация кандидатов
├── test_inference.py        # Unit: предсказание модели
├── test_schemas.py          # Unit: валидация входных данных (некорректный ввод)
├── test_api.py              # Integration: все эндпоинты
└── test_auth.py             # Integration: аутентификация X-API-Key
```

---

## test_config_generator.py — 19 тестов

Проверяет генератор случайных конфигураций (`recommender/config_generator.py`).

| Тест | Что проверяет |
|------|---------------|
| `test_returns_correct_count` | Возвращает ровно N кандидатов |
| `test_executor_cores_within_worker_cores` | `executor_cores ≤ worker_cores` |
| `test_executor_cores_positive` | `executor_cores ≥ 1` |
| `test_executor_memory_multiple_of_1g` | Память кратна 1024 МБ |
| `test_executor_memory_within_worker_memory` | Память не превышает RAM воркера |
| `test_executor_instances_within_workers` | `executor_instances ≤ workers` |
| `test_io_codec_valid` | Кодек только `lz4` или `snappy` |
| `test_boolean_fields_0_or_1` | Булевы параметры: только 0 или 1 |
| `test_memory_fraction_range` | `memory_fraction ∈ [0.3, 0.8]` |
| `test_constraint_max_executor_cores` | Ограничение `max_executor_cores` соблюдается |
| `test_constraint_max_executor_memory` | Ограничение `max_executor_memory_mb` соблюдается |
| `test_constraint_max_instances` | Ограничение `max_executor_instances` соблюдается |
| `test_rng_seed_reproducibility` | Одинаковый seed → одинаковые кандидаты |
| `test_different_seeds_differ` | Разные seeds → разные кандидаты |
| … и ещё 5 тестов диапазонов параметров | |

---

## test_inference.py — 12 тестов

Проверяет RF-суррогат (`recommender/inference.py`) напрямую, без HTTP.

| Класс | Тест | Что проверяет |
|-------|------|---------------|
| `TestPredictOne` | `test_returns_positive_prediction` | Предсказание > 0 |
| | `test_band_ordered` | p5 ≤ predicted ≤ p95 |
| | `test_no_nan` | Нет NaN в выводе |
| | `test_different_configs_differ` | Разные конфиги → разные предсказания |
| | `test_both_profiles_accepted` | `small` и `large` работают без ошибок |
| `TestPredictBatch` | `test_shape_matches_input` | Размер вывода совпадает с входом |
| | `test_all_positive` | Все предсказания > 0 |
| | `test_no_nan` | Нет NaN в батче |
| | `test_lows_leq_highs` | p5 ≤ p95 для всех строк |
| | `test_single_candidate_matches_predict_one` | Батч из 1 = predict_one |
| `TestDefaultRuntime` | `test_returns_positive` | default_runtime > 0 |
| | `test_large_faster_than_small_is_not_required` | Оба профиля работают без ошибок |

---

## test_schemas.py — 12 тестов (некорректный ввод)

Проверяет что API возвращает `422 Unprocessable Entity` на неверные данные.

| Тест | Что проверяет |
|------|---------------|
| `test_valid_minimal` | Минимальный корректный запрос → 200 |
| `test_missing_job_type` | Нет `job_type` → 422, ошибка содержит "job_type" |
| `test_missing_topology` | Нет `topology` → 422 |
| `test_missing_profile` | Нет `profile` → 422 |
| `test_topology_workers_zero` | `workers=0` → 422 |
| `test_topology_workers_negative` | `workers=-1` → 422 |
| `test_top_k_zero` | `return_top_k=0` → 422 |
| `test_top_k_above_max` | `return_top_k=100` → 422 |
| `test_unsupported_job_type_returns_422` | Неизвестный `job_type` → 422, сообщение "not supported" |
| `test_valid_minimal` (predict) | Минимальный корректный `/predict` → 200 |
| `test_missing_executor_cores` | Нет `executor_cores` → 422 |
| `test_executor_memory_below_min` | `executor_memory_mb=100` (< 256) → 422 |

---

## test_api.py — 29 интеграционных тестов

Полный HTTP-цикл через `httpx.ASGITransport` с реально загруженной PageRank-моделью.
В тестовой фикстуре `settings.n_candidates` снижается до 50, чтобы не гонять тяжёлый RandomForest по 200 кандидатам в каждом сценарии; runtime default остаётся 200.

| Класс | Тесты |
|-------|-------|
| `TestHealth` (3) | `status=ok`, `models_loaded` непустой, `model_dir` присутствует |
| `TestJobs` (3) | 200, `pagerank` в `supported`, `model_ready` непустой |
| `TestMetrics` (3) | 200, MAE и R² присутствуют, `dataset_version` непустой |
| `TestPredict` (4) | Happy path, предупреждение об `executor_cores`, предупреждение о `executor_memory`, `model_type` в ответе |
| `TestRecommend` (11) | Happy path, ранги последовательны, сортировка по runtime, speedup > 0, band упорядочена, top_k=1/2/5, нет дублей конфигов, model_info содержит job_type/mae/r2, constraint по cores/mem/instances |
| `TestHistory` (4) | GET /history возвращает список, GET /history/stats OK, /predict логируется, /recommend логируется |
| `TestUI` (1) | GET / → 200, Content-Type: text/html, содержит "Spark Config Recommender" |

---

## test_auth.py — 7 тестов

| Тест | Что проверяет |
|------|---------------|
| `test_no_key_allowed_when_auth_disabled` | Без API_KEY: запрос без ключа → 200 |
| `test_any_key_allowed_when_auth_disabled` | Без API_KEY: запрос с любым ключом → 200 |
| `test_valid_key_allowed` | С API_KEY: правильный ключ → 200 |
| `test_wrong_key_rejected` | С API_KEY: неверный ключ → 401 |
| `test_public_endpoint_accessible_without_key` | Публичный `/recommend` доступен без ключа даже при включённом API_KEY |
| `test_admin_endpoint_requires_key` | `/admin/models` без ключа → 403 |
| `test_admin_endpoint_valid_key` | `/admin/models` с правильным ключом → 200 |

---

## Запуск

```bash
# Все тесты
.venv/bin/pytest -v

# С покрытием
.venv/bin/pytest --cov=recommender --cov-report=term-missing
```

---

## Что НЕ покрыто автотестами

| Сценарий | Причина | Как проверять |
|----------|---------|---------------|
| E2E: predicted vs фактический Spark | Требует живой кластер | Вручную: запустить HiBench с рекомендованным конфигом |
| Нагрузочный тест (p95 < 1500 мс) | Нужен `locust` | `locust -f scripts/locustfile.py` |
| Стабильность рекомендаций | Не детерминировано | `python scripts/test_stability.py` |
