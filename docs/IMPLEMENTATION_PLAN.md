# План реализации: Spark Config Recommender

Живой документ. Обновляется по мере выполнения работ.
Последнее обновление: 2026-04-27.

---

## Легенда

- ✅ Готово
- 🔄 В процессе / частично
- ⬜ Не начато (можно начать)
- ❌ Заблокировано (ждёт зависимости)
- ❓ Под вопросом / требует уточнения

---

## 1. Инфраструктура и окружение

| # | Задача | Статус | Примечание |
|---|--------|--------|------------|
| 1.1 | Docker-окружение: Spark Standalone + HDFS + HiBench + MLflow | ✅ | `docker-compose.yml` |
| 1.2 | Образы: hadoop, spark, hibench, mlflow | ✅ | push в Docker Hub |
| 1.3 | Образ: recommender (FastAPI-сервис, multi-stage) | ✅ | `docker/recommender/Dockerfile` |
| 1.4 | Сервис recommender в docker-compose (порт 8001, volume ./out) | ✅ | |
| 1.5 | NodeManager zombie-fix (`init: true`) | ✅ | Добавлен в docker-compose с комментарием. На VM применить при следующем перезапуске NM. |

> **Архитектурная заметка:** Spark работает в Standalone-режиме (`spark://spark-master:7077`).
> YARN NodeManagers участвуют **только** в prepare-фазе HiBench (MapReduce). Spark executor'ами
> управляет Spark Master, не YARN. `init: true` решает проблему зомби-процессов от MapReduce-
> контейнеров (NM-JVM как PID 1 не вызывает wait() → 20k+ зомби → CPU-шторм в мониторинге).
> CPU-лимит NM **не устанавливается** — он замедлил бы prepare, не устранив зомби.

---

## 2. Сбор датасета

| # | Задача | Статус | Примечание |
|---|--------|--------|------------|
| 2.1 | WordCount — 3 топологии × 2 профиля | ✅ | `data/wc_train_merged.csv`, 401 строка |
| 2.2 | PageRank — 6 топологий × 2 профиля × 80 конфигов | ✅ | `data/hibench_train_20260424_175032.csv` (960 строк) |
| 2.3 | Фильтрация шума PageRank (CV > 0.20) | ✅ | `data/hibench_train_20260424_175032_clean.csv` (917 строк, −43). Проблемные ячейки: 6×6×12/small (63/80) и 8×6×12/small (66/80) — шум от NM-зомби на коротких задачах |
| 2.4 | TeraSort + WordCount — расширенный мульти-нагрузочный сбор | 🔄 | Следующий сбор на VM; PageRank уже собран и очищен |
| 2.5 | KMeans | ⬜ | Опционально, зависит от бюджета VM |
| 2.6 | Агрегация всех нагрузок в единый датасет + финальная валидация | ❌ | Ждёт завершения п.2.4 |
| 2.7 | Снэпшот входных данных HDFS | ✅ | `scripts/snapshot_hdfs_input.sh` |

---

## 3. REST API сервис (`recommender/`)

| # | Задача | Статус | Примечание |
|---|--------|--------|------------|
| 3.1 | `GET /health` | ✅ | |
| 3.2 | `GET /jobs` | ✅ | |
| 3.3 | `GET /metrics` | ✅ | MAE/RMSE/R²/MAPE из `report.json` |
| 3.4 | `POST /predict` | ✅ | Конфиг → время + confidence band (p5/p95 деревьев RF) |
| 3.5 | `POST /recommend` | ✅ | Random sampling → RF-суррогат → top-K + speedup vs default |
| 3.6 | Веб-интерфейс (форма `/`) | ✅ | `recommender/static/index.html`, vanilla JS, тёмная тема |
| 3.7 | История запросов (SQLite) | ✅ | `recommender/history.py`, эндпоинты `GET /history`, `GET /history/stats` |
| 3.8 | Двухуровневая аутентификация (X-API-Key) | ✅ | `recommender/auth.py`. user=публичные эндпоинты без ключа, admin=`/admin/*` требует ключ. Включается через `RECOMMENDER_API_KEY` env |
| 3.9 | Admin-эндпоинты (`GET /admin/models`, `POST /admin/models/reload`) | ✅ | Перезагрузка модели без рестарта сервиса. Защищены `require_admin` |
| 3.10 | Admin-панель в Web UI | ✅ | Кнопка «Войти как администратор» в хедере → модальное окно → вкладка «Управление моделями» |
| 3.11 | Optimizer: Bayesian (Optuna TPE) вместо random sampling | ⬜ | Улучшение `/recommend` |
| 3.12 | Optimizer: Offline DQN | ❌ | Ждёт п.5.5 |
| 3.13 | Optimizer: Q-learning (Sensors-22) | ❌ | Ждёт п.5.6 |

---

## 4. Тесты и покрытие

| # | Задача | Статус | Примечание |
|---|--------|--------|------------|
| 4.1 | Unit: `config_generator` — диапазоны, constraints, воспроизводимость | ✅ | 19 тестов |
| 4.2 | Unit: `inference.predict_one` / `predict_batch` — форма, не NaN, позитивность | ✅ | 12 тестов |
| 4.3 | Unit: Pydantic-схемы — невалидный ввод → 422 | ✅ | 12 тестов |
| 4.4 | Integration: все 5 эндпоинтов + history + UI | ✅ | 29 тестов |
| 4.5 | Integration: `/recommend` с жёсткими constraints + auth | ✅ | 7 тестов auth (включая admin-эндпоинты и двухуровневую проверку ролей) |
| 4.6 | Покрытие ≥ 80% (`pytest --cov recommender/`) | ✅ | **93%**, 79 тестов, 0 failed |
| 4.7 | E2E: predicted vs actual (реальный Spark, PageRank) | 🔄 | Скрипт `scripts/e2e_validate.sh` готов. Требует запущенный docker compose (hibench + spark-master + HDFS). Топология: `WORKERS=4 CORES=2 RAM_GB=4` (из зоны обучения). |
| 4.8 | Нагрузочный тест (`locust`): p50 < 500 ms, p95 < 1500 ms | ⬜ | Ручной запуск |

---

## 5. Обучение моделей

| # | Задача | Статус | Примечание |
|---|--------|--------|------------|
| 5.1 | PageRank baseline на очищенном датасете | ✅ | `out/final_best/pagerank`, RandomForest_RandomSearch, MAE≈8.97s |
| 5.2 | Переобучение baseline на мульти-нагрузочном датасете с `job_type` | ❌ | Ждёт TeraSort/KMeans/WordCount v2 |
| 5.3 | Сравнение: RF, LightGBM, CatBoost, XGBoost (кросс-валидация) | ❌ | Ждёт мульти-нагрузочный датасет |
| 5.4 | Multi-task (одна модель) vs per-workload | ❌ | Ждёт мульти-нагрузочный датасет |
| 5.5 | Offline DQN на новом датасете | ❌ | Ждёт мульти-нагрузочный датасет |
| 5.6 | Improved Q-learning (Sensors-22) | ❌ | Ждёт мульти-нагрузочный датасет |
| 5.7 | Графики: MAE по job_type, predicted vs actual scatter | 🔄 | PageRank графики готовы; общий график ждёт остальные нагрузки |

---

## 6. Документация и отчёт

| # | Задача | Статус | Примечание |
|---|--------|--------|------------|
| 6.1 | UML-диаграммы (5 листов в .drawio + SVG export) | ✅ | `docs/diagrams/spark_recommender_diagrams.drawio`, `docs/diagrams/*.svg` |
| 6.2 | Глава 1 (проектирование) | ✅ | Черновик в .docx |
| 6.3 | Глава 2 (датасет) | 🔄 | Заглушки → заменить на фактические числа после п.2.5 |
| 6.4 | Глава 3 (модели) | 🔄 | WordCount + PageRank числа готовы; мульти-нагрузочный итог ждёт п.5.2 |
| 6.5 | Глава 4 (REST API) — листинги, скриншот /docs | 🔄 | Реализация готова; нужен скриншот и пример ответа |
| 6.6 | Глава 5 (тестирование) | 🔄 | Unit/integration готовы; E2E и locust ещё нет |
| 6.7 | Глава 6 (документация): README, ссылка на репо | 🔄 | README и runbook актуализированы; нужен финальный commit/tag |
| 6.8 | Заключение — итоговые числа | ❌ | Ждёт всего остального |

---

## Критический путь

```
2.4 Завершение TeraSort/WordCount v2
 └─► 2.6 Единый мульти-нагрузочный датасет
      └─► 5.2 Переобучение с job_type
           ├─► 5.5/5.6 DQN + Q-learning → 3.10/3.11 Optimizer в API
           │    └─► 4.7 E2E тесты
           └─► 5.7 Графики → 6.4 Глава 3 → 6.8 Заключение
```

---

## Текущий приоритет (2026-04-27)

1. **🔄 4.7** — запустить `scripts/e2e_validate.sh` (WORKERS=4 CORES=2 RAM_GB=4) для E2E predicted vs actual. Скрипт готов, нужен docker compose up.
2. **🔄 2.4** — добрать TeraSort/WordCount v2 на VM.
3. **❌ 2.6 → 5.2** — собрать единый датасет и переобучить модель с `job_type`.
4. **🔄 6.4–6.6** — перенести фактические PageRank/API/test-числа в отчёт (скриншот /docs, пример ответа /recommend, результат E2E).
