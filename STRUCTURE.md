# Структура проекта

## Корневые файлы

| Файл | Описание |
|------|----------|
| `README.md` | Главная документация проекта: цели, архитектура, методы ML, инструкции запуска |
| `STRUCTURE.md` | Описание структуры проекта и назначения файлов (этот файл) |
| `docker-compose.yml` | Оркестрация Docker-контейнеров: Spark master/workers, HiBench, MLflow, HDFS |
| `.gitignore` | Исключения для Git |

---

## config/ — Конфигурационные файлы

### config/hadoop/
| Файл | Описание |
|------|----------|
| `core-site.xml` | Базовая конфигурация Hadoop (HDFS endpoint) |
| `hdfs-site.xml` | Настройки HDFS (репликация, пути) |
| `yarn-site.xml` | Конфигурация YARN Resource Manager |
| `mapred-site.xml` | Настройки MapReduce |
| `capacity-scheduler.xml` | Конфигурация планировщика очередей |
| `log4j.properties` | Настройки логирования Hadoop |
| `hadoop.env` | Переменные окружения для Hadoop-контейнеров |

### config/hibench/
| Файл | Описание |
|------|----------|
| `hibench.conf` | Основная конфигурация HiBench: профили нагрузки (tiny/small/large/huge) |
| `spark.conf` | Параметры Spark для HiBench: master URL, память, ядра |

### config/spark/
| Файл | Описание |
|------|----------|
| `jmx_exporter_config.yml` | Конфигурация JMX Exporter для Prometheus-метрик Spark |

### config/prometheus/
| Файл | Описание |
|------|----------|
| `prometheus.yml` | Конфигурация Prometheus: targets для сбора метрик |

---

## docker/ — Docker-образы

### docker/hibench/
| Файл | Описание |
|------|----------|
| `Dockerfile` | Образ HiBench: Ubuntu + Java + Hadoop + Spark + HiBench benchmark suite |
| `maven-settings.xml` | Настройки Maven для сборки HiBench |
| `fix/` | Python-патчи для исправления багов HiBench |

### docker/mlflow/
| Файл | Описание |
|------|----------|
| `Dockerfile` | Образ MLflow: tracking server для логирования экспериментов |

### docker/spark/
| Файл | Описание |
|------|----------|
| `Dockerfile` | Кастомный образ Spark с JMX Exporter |
| `start-history-server.sh` | Скрипт запуска Spark History Server |

---

## scripts/ — Скрипты сбора данных

| Файл | Описание |
|------|----------|
| `run_wordcount_experiments.sh` | **Главный оркестратор**: запускает сбор датасета по всем топологиям кластера. Управляет Docker-воркерами, архивирует старые данные. |
| `collect_wordcount_data.sh` | **Сборщик данных**: выполняется внутри HiBench-контейнера. Генерирует случайные конфигурации Spark, запускает бенчмарк, вычисляет медиану времени. |

### scripts/utils/
| Файл | Описание |
|------|----------|
| `check_cluster.sh` | Проверка состояния Spark-кластера |
| `up_all_and_check.sh` | Запуск всех контейнеров и проверка готовности |
| `hibench_auto_fix.sh` | Автоматическое исправление проблем HiBench |
| `run_wordcount_pipeline.sh` | Устаревший пайплайн (см. archive/) |

---

## training/ — ML-код обучения моделей

| Файл | Описание |
|------|----------|
| `train_baseline_mlflow.py` | Baseline модели: DummyRegressor, RandomForest, MLP + логирование в MLflow |
| `train_dnn_qlearning.py` | DNN предиктор + Q-Learning оптимизатор (первая версия) |
| `train_dnn_qlearning_v3.py` | Улучшенная версия: RandomForest + Simulated Annealing + Q-Learning |
| `collect_agg.py` | Утилита агрегации CSV-файлов с результатами |
| `requirements.txt` | Python-зависимости для обучения |

---

## data/ — Датасеты

| Файл | Описание |
|------|----------|
| `wc_train_all.csv` | Основной датасет WordCount (собирается скриптами) |
| `hibench.report` | Сырой отчёт HiBench (все запуски) |

### data/snapshots/
Архив промежуточных версий датасета:
| Файл | Описание |
|------|----------|
| `wc_train_YYYYMMDD_HHMMSS.csv` | Автоматические бэкапы при перезапуске сбора |
| `wc_current_snapshot.csv` | Текущий снимок для анализа |
| `wordcount_agg.csv` | Агрегированные данные (старый формат) |
| `wordcount_runs.csv` | Детальные прогоны (старый формат) |

---

## models/ — Сохранённые модели

### models/rf_baseline/
| Файл | Описание |
|------|----------|
| `model_rf.joblib` | Обученная модель RandomForest |
| `preprocess.joblib` | Пайплайн препроцессинга (OneHotEncoder + StandardScaler) |
| `report.json` | Метрики модели: MAE, RMSE, MAPE, R² |
| `qlearning_history_*.csv` | История обучения Q-Learning агента |
| `qlearning_suggestions.csv` | Рекомендованные конфигурации |
| `plots/` | Графики: предсказания vs реальность, траектории Q-Learning |

---

## Формат датасета (wc_train_all.csv)

22 колонки:

**Топология кластера (3):**
- `topology_workers` — количество воркеров
- `topology_worker_cores` — ядер на воркер
- `topology_worker_mem_gb` — память воркера (GB)

**Параметры Spark (16):**
- `executor_cores`, `executor_memory`, `executor_instances`
- `driver_cores`, `driver_memory`
- `memory_fraction`, `memory_storageFraction`
- `shuffle_compress`, `spill_compress`, `shuffle_file_buffer`
- `broadcast_block`, `broadcast_compress`
- `maxSizeInFlight`, `io_codec`, `rpc_message_maxSize`, `rdd_compress`

**Метаданные (1):**
- `profile` — профиль нагрузки (large)

**Целевая переменная (1):**
- `median_duration_s` — медиана времени выполнения (секунды)

**Служебные (1):**
- `exit_code` — код завершения (0 = успех)
