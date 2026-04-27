# План реализации REST API и тестирования

Этот документ начинался как рабочий план API и тестирования. На 2026-04-27 большая часть плана реализована в `recommender/`, `docker/recommender/Dockerfile`, `requirements-prod.txt`, `tests/` и `docs/RUNBOOK.md`; ниже оставлены исходные проектные решения и пункты, которые ещё полезны для отчёта.

Дата старта: 2026-04-25.

---

## 1. REST API: что строим

### 1.1. Контракт сервиса

Эндпоинты:

- `GET /health` — лайвнесс-проба, проверка что модель загружена;
- `GET /jobs` — список поддерживаемых нагрузок (сейчас runtime-модель есть только для `pagerank`);
- `GET /metrics` — метрики качества предикторов (MAE, RMSE, R², MAPE) по каждой загруженной нагрузке + версия датасета;
- `POST /predict` — предсказать время выполнения для **уже заданной** Spark-конфигурации (нужно для интеграционных тестов и быстрой ручной проверки);
- `POST /recommend` — основной эндпоинт. На вход — `job_type`, `input`, `topology`, опционально `constraints` и `preferences`. На выход — top-K рекомендаций с предсказанным временем, ускорением vs default и предупреждениями.

Контракт уже зафиксирован в `docs/VKR_PLAN.md` разделе A.3 (входной/выходной JSON). Он не требует пересмотра — отчёт про §4 пишется по нему.

### 1.2. Стек

**База — Python.** FastAPI + Pydantic + Uvicorn. Причины:
- модель и препроцессор уже на Python (scikit-learn, PyTorch, Optuna, stable-baselines3) — переписывать на C++ значит переписывать всё;
- ColumnTransformer с OneHotEncoder, StandardScaler, log1p-таргетом — scikit-learn;
- MLflow Model Registry — нативно Python;
- латентность одного запроса всё равно в сотни мс (sklearn predict + ConfigGenerator из 200 кандидатов + сортировка), узкое место — ML-инференс, не транспорт.

**По C++ для самого сервиса.** Это технически возможный, но в нашем кейсе плохо окупаемый путь. Чтобы перенести инференс RF в C++, надо либо:
- экспортировать дерево в формат типа `treelite` / `lleaves` и вызывать его через pybind11. Latency: 0.1 мс vs 2–5 мс sklearn. Стоимость: + библиотека, + сборка, + риск рассинхронизации с обучающим pipeline. Окупается только при тысячах rps;
- или экспортировать в ONNX (skl2onnx → onnxruntime). Это дешевле, но всё равно усложняет деплой ради 2 мс при единичных запросах.

**Предлагаемое решение:** оставляем Python, но точку, где это окупится позже, помечаем явно — RL-инференс на DQN/PPO. Если на проде агент должен дёргаться часто (online-recommendation), его можно вынести в C++/Rust сервис с `onnxruntime` за тем же FastAPI-фасадом. На текущем этапе такого нагрузочного профиля нет.

**Контейнеризация:** Dockerfile в `docker/recommender/Dockerfile`, multi-stage сборка (slim-python + только runtime-зависимости из `requirements-prod.txt`), не тянем torch/optuna/stable-baselines3 в образ — они нужны только в обучении.

### 1.3. Структура кода

```
recommender/
├── __init__.py
├── api.py                 # FastAPI app, роутер
├── schemas.py             # Pydantic-модели запросов и ответов
├── inference.py           # PredictorService: загрузка модели + predict
├── config_generator.py    # ConfigGenerator: LHS-сэмплинг кандидатов
├── model_registry.py      # disk-loader моделей, выбор лучшей по MAE из report.json
├── history.py             # SQLite-история запросов
├── settings.py            # pydantic-settings: пути, версии моделей, лимиты
├── tests/                 # см. раздел 2
└── Dockerfile
```

### 1.4. Версионирование моделей

`out/final_best/<job_type>/` — фиксированный артефакт с моделью + препроцессором + `report.json`:

```json
{
  "model_id": "rf_sa_terasort_v2_20260420",
  "predictor_type": "RandomForest_SimulatedAnnealing",
  "trained_on_dataset": "hibench_train_merged_20260418.csv",
  "schema_version": 2,
  "metrics": {"mae_s": 0.91, "rmse_s": 1.42, "r2": 0.94, "mape_pct": 4.2},
  "feature_columns": ["job_type", "topology_workers", ...],
  "git_sha": "abc1234"
}
```

Сервис на старте читает этот JSON и публикует через `/metrics`. При смене модели — переписали папку, перезапустили контейнер, ничего больше делать не надо. Альтернатива — MLflow Model Registry, но для команды из одного человека он дорог по поддержке.

### 1.5. Что осталось решить

- Нужен ли `/upload-dataset` или другой write-эндпоинт. По текущей постановке нет: датасет собирается отдельно на VM, сервис только инференс. Если потом понадобится online-fine-tuning — добавим как отдельный сервис, не в этот.
- Аутентификация. Внутрь компании достаточно `X-API-Key` через `Depends`. Для публичного — Keycloak/JWT, но это вне scope.
- Rate-limit. Если решим что нужен — `slowapi` на FastAPI, тривиально.

---

## 2. Тесты: что покрываем и как

### 2.1. Уровни тестирования

**Unit-тесты** (быстрые, без Docker, в обычном CI):
- `ConfigGenerator` — LHS возвращает требуемое число конфигов, все они укладываются в `param_grid`, нарушения `constraints` отфильтровываются;
- `PredictorService.predict_with_band` — на синтетических данных возвращает массив правильной формы, не падает на NaN после препроцессинга;
- `OptimizerService.select_top_k` — корректно сортирует по runtime, не дублирует конфигурации;
- `schemas` — Pydantic ловит невалидный JSON (отрицательный размер, неизвестный `job_type`).

Минимум 80% покрытия по `recommender/`. Запуск: `pytest -q` в обычном Python venv.

**Integration-тесты API** (без Spark, mock-предиктор):
- `httpx.AsyncClient` против поднятого in-process FastAPI приложения;
- проверяем happy-path `/recommend`, `/predict`, `/jobs`, `/metrics`, `/health`;
- проверяем невалидные входы → 422 с конкретным `loc` от Pydantic;
- проверяем что `/recommend` с жёсткими `constraints` (например `max_executor_cores=2`) возвращает только конфиги, удовлетворяющие лимиту.

**End-to-end на реальном Spark** (медленно, отдельный CI-job):
- поднимается мини-кластер через `docker-compose.test.yml` (см. §2.2);
- скрипт `tests/e2e/run_predicted_vs_actual.py`:
  1. дёргает `POST /recommend` с заранее подобранным запросом;
  2. берёт top-1 рекомендацию;
  3. формирует `spark.conf` и запускает HiBench с этой конфигурацией через collect-скрипт (но без сэмплинга — `USE_FIXED=1`);
  4. сравнивает `predicted_runtime_s` vs `actual median_duration_s`;
  5. проверяет, что отклонение в пределах допуска (например, ±25% — это валидация качества модели, не SLA сервиса).

### 2.2. Топология тестового Spark-кластера

Самое опасное место — случайно поднять в CI/локальной машине разработчика тяжёлый кластер и убить host. Поэтому для тестов:

```
docker-compose.test.yml — отдельный compose с явными лимитами:
  spark-master:    cpus: 1.0, memory: 1g
  spark-worker-1:  cpus: 1.0, memory: 1.5g  (executor_cores=1, executor_mem=1g)
  spark-worker-2:  cpus: 1.0, memory: 1.5g  (executor_cores=1, executor_mem=1g)
  hibench:         cpus: 1.0, memory: 1g
  namenode + datanode: cpus: 0.5, memory: 1g каждый
  (resourcemanager + nodemanager — НЕ поднимаем; в e2e тестах берём только
   workloads, у которых prepare обходится без MapReduce, т.е. WordCount и TeraSort.
   PageRank/KMeans тестим только в более тяжёлом nightly job.)
```

Итого тестовый кластер ест ~7 vCPU и ~8 GB RAM. Это укладывается в разработческий ноутбук (и в типичный self-hosted runner).

**Жёсткие safeguards в коде тестов:**

```python
# tests/e2e/conftest.py
def assert_test_topology():
    """Не запускаем тест, если контейнеры не из docker-compose.test.yml."""
    info = subprocess.check_output(["docker", "compose", "ls"]).decode()
    assert "spark-test" in info, (
        "E2E тест требует docker-compose -f docker-compose.test.yml up -d. "
        "Не запускайте против обычного docker-compose.yml — "
        "это убьёт ноутбук разработчика."
    )

def assert_total_resources_within_limit(max_cores: int = 8, max_mem_gb: int = 12):
    """Считаем суммарные cpu/memory лимиты во всех контейнерах compose. Если
    > порога — фейлим тест ДО запуска Spark-задачи."""
    ...
```

Это не теоретический риск — на 64-ядерной VM сбора такая защита не нужна, но локально или на маленьком CI-runner-е без неё легко выкатить тест который потащит за собой 30 GB RAM и зависнет.

**Тестовые данные:** WordCount input в HDFS — 10 МБ синтетических данных, генерируется в `pytest fixture` один раз и кэшируется в volume `hdfs-test-nn`. Это даёт прогон одной Spark-задачи в 5-10 секунд на тестовом кластере, что приемлемо для CI.

### 2.3. Что точно НЕ делаем в тестах

- Не запускаем PageRank/KMeans на dev-машине — слишком дорого;
- Не запускаем полный сбор датасета (TARGET_SAMPLES > 5);
- Не пытаемся в e2e-тесте измерить точный speedup vs default — это работа отдельного валидационного прогона в отчёте, не CI-теста.

### 2.4. Нагрузочные тесты API

`locust` сценарий с одним пользователем, шлёт `POST /recommend` подряд, измеряем p50/p95 латентность. Целевые числа на одной vCPU без батчинга:
- p50 < 500 ms;
- p95 < 1500 ms;
- error rate = 0.

Если не укладываемся — профилируем (`py-spy`), смотрим где время: PredictorService.predict, ConfigGenerator (LHS из 200 кандидатов), сериализация ответа. Скорее всего — в predict, и тогда есть путь сократить число кандидатов с 200 до 50.

Запуск нагрузочных тестов — НЕ в обычном CI. Отдельный manual job.

### 2.5. Acceptance criteria для отчёта по этапу 5–6

- [ ] `pytest -q` зелёный, покрытие ≥ 80%;
- [ ] `docker-compose -f docker-compose.test.yml up -d && pytest tests/e2e/` зелёный;
- [ ] хотя бы один прогон predicted vs actual записан в `data/validation_*.csv` для каждой нагрузки (TeraSort, PageRank, WordCount);
- [ ] `locust` отчёт за 5 минут с одним пользователем приложен;
- [ ] OpenAPI-документация автогенерится и доступна на `/docs`.

---

## 3. Что ещё надо сделать в репозитории

Не для тестов, а до защиты черновика, чтобы рассказ в отчёте про §4 был не пустым:

1. Скелет `recommender/api.py` — пусть даже только `/health` и `/jobs`. Это даст реальный код для листинга в Приложении.
2. Скелет `recommender/schemas.py` с Pydantic-классами `RecommendRequest`, `RecommendResponse` — листинг в Приложении.
3. `requirements-prod.txt` (отдельно от обучающего `requirements.txt`).
4. `recommender/Dockerfile` — multi-stage skeleton.
5. Пустые файлы тестов с docstring-описанием каждого теста (как «контракт»).
