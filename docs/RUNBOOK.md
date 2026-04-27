# Runbook: Spark Config Recommender — PageRank

Последнее обновление: 2026-04-27.  
Проверяется на локальной машине с Docker.

---

## 1. Быстрый старт

```bash
# Поднять кластер
docker compose up -d

# Подождать ~30 сек, затем проверить статус
docker compose ps
```

Все сервисы должны быть в состоянии `Up`:

| Сервис | URL |
|--------|-----|
| **Recommender UI** | http://localhost:8001/ |
| **Recommender API docs** | http://localhost:8001/docs |
| Spark Master | http://localhost:8080/ |
| HDFS NameNode | http://localhost:9870/ |
| MLflow | http://localhost:5000/ |

---

## 2. Автотесты

```bash
# Все 79 тестов — должны пройти
.venv/bin/pytest -v

# Покрытие — должно быть ≥ 90%
.venv/bin/pytest --cov=recommender --cov-report=term-missing
```

**Ожидаемый результат:** `79 passed, 0 failed`, покрытие около `93%`.

---

## 3. Проверка здоровья сервиса

```bash
curl -s localhost:8001/health | python3 -m json.tool
```

**Ожидаемое:**
```json
{
  "status": "ok",
  "models_loaded": ["pagerank"],
  "model_dir": "..."
}
```

Важно: `models_loaded` должен содержать `"pagerank"`.

---

## 4. Метрики обученной модели

```bash
curl -s localhost:8001/metrics | python3 -m json.tool
```

**Ожидаемые метрики (PageRank, test set, n=184):**

| Метрика | Значение |
|---------|---------|
| MAE | ≈ 8.97 s |
| RMSE | ≈ 16.51 s |
| R² | ≈ 0.844 |
| MAPE | ≈ 15.8% |

Датасет: `data/hibench_train_20260424_175032_clean.csv` (917 строк после фильтрации CV > 0.20).

---

## 5. Predict — предсказание для конкретного конфига

```bash
curl -s -X POST localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "pagerank",
    "profile": "large",
    "topology_workers": 4,
    "topology_worker_cores": 6,
    "topology_worker_mem_gb": 12,
    "executor_cores": 2,
    "executor_memory_mb": 4096,
    "executor_instances": 4,
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
    "io_codec": "lz4"
  }' | python3 -m json.tool
```

**Ожидаемое:**
- `predicted_runtime_s` > 0 (обычно 50–90 s для large/4×6×12)
- `confidence_band`: `[low, high]`, low ≤ predicted ≤ high
- `warnings`: пустой список (конфиг корректный)

---

## 6. Recommend — получить топ-3 рекомендации

```bash
curl -s -X POST localhost:8001/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "pagerank",
    "input": {"profile": "large"},
    "topology": {"workers": 4, "worker_cores": 6, "worker_memory_gb": 12},
    "preferences": {"return_top_k": 3}
  }' | python3 -m json.tool
```

**Ожидаемое:**
- 3 рекомендации, отсортированные по `predicted_runtime_s` (ascending)
- `predicted_speedup_vs_default` > 1.0 для всех
- `rank` идёт 1, 2, 3
- `model_info.predictor_mae_s` ≈ 8.97

**Проверка constraints:**
```bash
# Все executor_cores ≤ 2
curl -s -X POST localhost:8001/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "pagerank",
    "input": {"profile": "large"},
    "topology": {"workers": 4, "worker_cores": 6, "worker_memory_gb": 12},
    "constraints": {"max_executor_cores": 2},
    "preferences": {"return_top_k": 5}
  }' | python3 -c "import sys,json; recs=json.load(sys.stdin)['recommendations']; print('cores:', [r['config']['executor_cores'] for r in recs])"
```

**Проверка профиля small:**
```bash
curl -s -X POST localhost:8001/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "job_type": "pagerank",
    "input": {"profile": "small"},
    "topology": {"workers": 2, "worker_cores": 4, "worker_memory_gb": 8},
    "preferences": {"return_top_k": 3}
  }' | python3 -m json.tool
```
Для `small` профиля ожидается runtime 7–20 s (реальный диапазон в датасете 7.3–28 s).

---

## 7. Admin-эндпоинты

Админ-доступ всегда требует конкретный ключ: `RECOMMENDER_ADMIN_KEY`.
Если он не задан, используется fallback `RECOMMENDER_API_KEY`; если не задан ни
один из них, админ-эндпоинты закрыты.

```bash
# Список загруженных моделей с метриками
curl -s localhost:8001/admin/models \
  -H "X-API-Key: <ваш_ключ>" | python3 -m json.tool

# Перезагрузить модели с диска без рестарта сервиса
curl -s -X POST localhost:8001/admin/models/reload \
  -H "X-API-Key: <ваш_ключ>" | python3 -m json.tool

# Удалить модель из runtime-пула сервиса
curl -s -X DELETE localhost:8001/admin/models/pagerank/rf \
  -H "X-API-Key: <ваш_ключ>" | python3 -m json.tool
```

Публичные эндпоинты (`/recommend`, `/predict`, `/history`) всегда доступны без ключа.

---

## 8. E2E-валидация (реальный Spark)

Проверяет, насколько предсказанное время совпадает с фактическим прогоном HiBench PageRank.

**Предварительные условия:** `docker compose up -d` (hibench + spark-master + HDFS должны быть Up) и запущенный recommender.

```bash
# Минимальный запуск (безопасно для локальной машины)
WORKERS=2 CORES=2 RAM_GB=4 PROFILE=large bash scripts/e2e_validate.sh

# Репрезентативный — топология из зоны обучения датасета PageRank
WORKERS=4 CORES=2 RAM_GB=4 PROFILE=large bash scripts/e2e_validate.sh

# С увеличенным числом повторений для уменьшения дисперсии
WORKERS=4 CORES=2 RAM_GB=4 REPEATS=5 bash scripts/e2e_validate.sh
```

Скрипт сам:
1. Запрашивает топ-1 рекомендацию у `/recommend`
2. Запускает нужное число Spark-воркеров через `docker run`
3. Пишет `spark.conf` в контейнер hibench
4. Запускает `prepare.sh` (если входных данных ещё нет в HDFS)
5. Запускает `run.sh` `REPEATS` раз и читает фактическое время из `hibench.report`
6. Выводит сравнение predicted vs actual + MAPE + вывод о точности

**Ожидаемый вывод (MAPE ≤ 20% — зелёный свет):**
```
Предсказание:   62.3 с  (диапазон 45.1–81.2 с)
Факт:           [58.7, 61.2, 60.4]  →  среднее=60.1 с  медиана=60.4 с
Ошибка:         |факт−прогноз| = 2.2 с  (MAPE=3.6%)
В диапазоне p5–p95: Да ✓

ИТОГ: прогноз точен (MAPE=3.6% ≤ 20%, целевой порог)
```

---

## 9. История запросов

```bash
# Последние 10 запросов
curl -s "localhost:8001/history?limit=10" | python3 -m json.tool

# Статистика
curl -s localhost:8001/history/stats | python3 -m json.tool
```

**Ожидаемое в stats:**
- `total_requests` > 0 после п.5–6
- `by_endpoint`: `/predict` и `/recommend` присутствуют
- `avg_duration_ms` < 500

---

## 10. Web UI

1. Открыть http://localhost:8001/ — должна загрузиться страница с тёмной темой
2. Выбрать `Job Type: pagerank`, `Profile: large`
3. Ввести топологию: Workers=4, Cores=6, RAM=12
4. Нажать **Get Recommendations**
5. Должны появиться 3 карточки с rank, runtime, speedup×, confidence band
6. Раскрыть **Constraints**, выставить `Max executor cores = 2` → перезапросить → все карточки должны показывать `executor_cores ≤ 2`
7. Переключить `Profile: small` → числа должны уменьшиться (small задачи быстрее)

---

## 11. Проверка модели напрямую

```bash
.venv/bin/python -c "
from pathlib import Path
from recommender.model_registry import ModelRegistry
from recommender.inference import PredictorService

reg = ModelRegistry(Path('out/final_best'))
reg.load_all(['pagerank'])
svc = PredictorService(reg.get('pagerank'))

# Предсказание одного конфига
row = {
    'topology_workers': 4, 'topology_worker_cores': 6, 'topology_worker_mem_gb': 12,
    'profile': 'large', 'executor_cores': 2, 'executor_memory_mb': 4096,
    'executor_instances': 4, 'driver_cores': 1, 'driver_memory_mb': 1024,
    'memory_fraction': 0.6, 'memory_storageFraction': 0.5,
    'shuffle_compress': 1, 'spill_compress': 1, 'shuffle_file_buffer_kb': 32,
    'broadcast_block_mb': 4, 'broadcast_compress': 1,
    'maxSizeInFlight_mb': 48, 'rpc_message_maxSize': 128,
    'rdd_compress': 0, 'io_codec': 'lz4',
}
pred, lo, hi = svc.predict_one(row)
print(f'Predicted: {pred:.1f}s  band=[{lo:.1f}, {hi:.1f}]')
print(f'Default runtime (4×6×12/large): {svc.default_runtime(4, 6, 12, \"large\"):.1f}s')
"
```

---

## 12. Графики для отчёта

Все графики PageRank-модели находятся в `out/final_best/pagerank/plots/`:

| Файл | Содержание |
|------|-----------|
| `bar_mae.png` | MAE всех моделей (сравнение) |
| `bar_r2.png` | R² всех моделей |
| `bar_mape.png` | MAPE всех моделей |
| `bar_rmse.png` | RMSE всех моделей |
| `scatter_randomforest_randomsearch.png` | Predicted vs Actual (лучшая модель) |
| `sa_convergence.png` | Сходимость Simulated Annealing |

---

## Чеклист ✓

- [ ] `docker compose ps` — все сервисы Up
- [ ] `GET /health` → `models_loaded: ["pagerank"]`
- [ ] `GET /metrics` → pagerank MAE ≈ 8.97 s, R² ≈ 0.844
- [ ] `POST /predict` → `predicted_runtime_s` > 0, `confidence_band` упорядочена
- [ ] `POST /recommend` → 3 рекомендации, speedup > 1.0, ранги 1–3
- [ ] `POST /recommend` с constraint → cores ≤ max_executor_cores
- [ ] `GET /history/stats` → `total_requests` > 0 после запросов
- [ ] Web UI открывается, рекомендации отображаются в карточках; вкладки «История», «Управление моделями» работают
- [ ] `pytest -v` → 79 passed, 0 failed
- [ ] `pytest --cov=recommender` → ≥ 90% (текущее ≈ 93%)
- [ ] E2E: `WORKERS=4 CORES=2 RAM_GB=4 bash scripts/e2e_validate.sh` → MAPE ≤ 20%
- [ ] Stability: `python scripts/test_stability.py` → CV < 5% (все 16 параметров)

---

## Известные ограничения

- **job_type=pagerank** — единственная нагрузка с собственной моделью. wordcount/terasort/kmeans временно отключены (модели для них не валидированы).
- **E2E тест** требует запущенного `docker compose` (hibench + spark-master + HDFS). Скрипт: `scripts/e2e_validate.sh`. Дефолтная топология `2×2×4`, репрезентативная `4×2×4`.
- **Confidence band** — p5/p95 деревьев RF, не строгий доверительный интервал. Для large-профиля band шире (~40–100 s).
