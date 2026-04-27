# Dataset Collection — Status and Next Steps

Дата: 2026-04-24.

Документ кратко описывает, в каком состоянии находится сбор датасета Spark/HiBench
после диагностики и рефакторинга.

## 1. Что было сломано

### 1.1. PageRank и KMeans

Prepare падал на `run_hadoop_job $DATATOOLS HiBench.DataGen ...` с сообщением:

```
JAR does not exist or is not a normal file: /opt/hibench/HiBench.DataGen
```

**Корневая причина:** образ `sertitanius/spark_machine_learning-hibench:1.0.0` поставляется
с неполным `/opt/hibench/conf/hibench.conf`: в нём отсутствует ключ
`hibench.hibench.datatool.dir`, из-за чего переменная `DATATOOLS` внутри
`run_hadoop_job` резолвится в пустую строку и Hadoop ищет несуществующий jar.

HiBench datatool-jar при этом присутствует в образе:
`/opt/hibench/autogen/target/autogen-8.0-SNAPSHOT-jar-with-dependencies.jar`.

Все нагрузки, которые используют `DATATOOLS`, падали по той же причине:
PageRank, KMeans, Bayes, SQL (aggregation/join/scan), streaming/wordcount seed.

### 1.2. TeraSort и WordCount

Работали без ошибок, потому что их `prepare/prepare.sh` вызывает
`HADOOP_EXAMPLES_JAR` (`teragen` и `randomtextwriter`), а не datatool-jar.

### 1.3. YARN NodeManager-ы

В ранбуке раньше стояла команда:

```bash
docker compose up -d namenode datanode resourcemanager spark-master spark-history hibench mlflow
```

В ней **нет** `nodemanager-1` и `nodemanager-2`. PageRank / KMeans / TeraSort
генерируют вход через MapReduce, поэтому без NM-ов MR-job висит в состоянии
`ACCEPTED` и prepare не прогрессирует.

### 1.4. Дубликаты prepare

Если запустить `prepare.sh` параллельно (например, случайно из нескольких
терминалов), несколько MR-job-ов лезут в один и тот же HDFS-путь и ломают друг
другу вывод (видно в ранних логах, где три pagerank-MR сидели в очереди и часть
падала с `Job failed!` уже после того, как первая успешно записала часть данных).

## 2. Что исправлено в репо

### 2.1. `config/hibench/hibench.conf`

Переписан так, чтобы быть каноничной конфигурацией HiBench. Содержит
`hibench.hibench.datatool.dir`, `hibench.home`, формат Sequence, default-имена
Input/Output и базовые параллельности. Смонтирован как
`/opt/hibench/conf/hibench.conf.template:ro` (на случай, если мы захотим позже
заменить ещё и in-container conf).

### 2.2. `scripts/collect_hibench_data.sh`

- Добавлен блок `ensure_kv` — при старте коллектор идемпотентно дописывает
  в `/opt/hibench/conf/hibench.conf` недостающие ключи: `hibench.home`,
  `hibench.hibench.datatool.dir`, `sparkbench.{in,out}putformat`,
  `hibench.workload.dir.name.{input,output}`, `hibench.masters/slaves.hostnames`.
- `hibench.scale.profile` пишется отдельно — если ключ отсутствует, добавляется.
- Prepare теперь **пропускается**, если HDFS-вход уже существует
  (`FORCE_PREPARE=1` отключает skip). Это экономит минуты на каждой ячейке
  `job_type x profile x topology`.

### 2.3. `scripts/smoke_hibench_workloads.sh`

- Self-heal hibench.conf тем же набором ключей.
- Preflight-проверка: если `yarn node -list` не вернул ни одного RUNNING NM,
  скрипт выходит с инструкцией, как их поднять.

### 2.4. `scripts/run_hibench_experiments.sh`

- Preflight-проверка YARN NodeManager-ов добавлена до прогона.

### 2.5. Валидация фикса

Прогон smoke_hibench_workloads.sh после фикса:

```
pagerank,small,1, prepare_rc=0, run_rc=0, duration=26s
kmeans,small,1,   prepare_rc=0, run_rc=0, duration=35s
```

Раньше pagerank возвращал prepare_rc=255, kmeans — в тех же условиях падал по
датагенератору.

## 3. Скрипты после консолидации

| Скрипт | Роль | Где запускается |
|---|---|---|
| `scripts/smoke_hibench_workloads.sh` | smoke test 3 нагрузок x N профилей; одна фиксированная топология; проверка prepare/run | host |
| `scripts/run_hibench_experiments.sh` | обход матрицы `workloads x profiles x topologies`; поднимает Spark workers под каждую топологию; зовёт collector внутри контейнера | host |
| `scripts/collect_hibench_data.sh` | генератор случайных валидных Spark-конфигов и сбор runtime на REPEATS повторов; пишет расширенный CSV | внутри контейнера `hibench` |
| `scripts/vm_preflight.sh` | проверка CPU/RAM/disk на новой VM | host |
| `scripts/snapshot_hdfs_input.sh` | архивация HDFS input в tar.gz | host |
| `scripts/restore_hdfs_input.sh` | восстановление HDFS input из tar.gz | host |
| `scripts/monitor_collection_resources.sh` | периодический snapshot CPU/RAM/диска во время сбора | host |
| `scripts/plot_combined.py` | графики по результатам | host |

Старые WordCount-специфичные скрипты перемещены в
`archive/legacy_scripts/`:

- `collect_wordcount_data.sh` (коллектор для одной нагрузки),
- `run_wordcount_experiments.sh` (host runner для одной нагрузки),
- `run_specific_config.sh` (валидация конкретной рекомендованной конфигурации
  поверх WordCount collector — нужно позже переписать под generic collector).

## 4. Рекомендуемый порядок запуска на VM

```bash
# 1. Docker + compose plugin должны быть установлены; см. docs/VM_DATASET_COLLECTION_RUNBOOK.md
# `docker compose up -d` без аргументов поднимает все сервисы, включая
# nodemanager-1/2 — это обязательно для PageRank/KMeans/TeraSort.
docker compose up -d

# 2. Smoke — 5-10 минут
WORKLOADS=wordcount,pagerank,kmeans,terasort \
PROFILES=small,large \
REPEATS=1 \
./scripts/smoke_hibench_workloads.sh

# 3. Посмотреть data/smoke_hibench_workloads_<timestamp>.csv.
#    Критерий: prepare_rc=0, run_rc=0, разумный duration на all cells.

# 4. Полный сбор
WORKLOADS=wordcount,pagerank,kmeans,terasort \
PROFILES=small,large \
TARGET_SAMPLES=20 \
REPEATS=5 \
./scripts/run_hibench_experiments.sh
```

## 5. Что НЕ делать

- Не запускать `prepare.sh` параллельно для одного и того же workload+profile —
  MR-job-ы подерутся за HDFS-путь и datagen частично сломается.
- Не выключать одновременно all NM-ы посреди сбора: любые in-flight
  prepare MR-job-ы повиснут.
- Не запускать обучение моделей на той же VM во время активного сбора —
  время выполнения Spark job-ов станет шумным.

## 6. Что ещё можно улучшить в collect_hibench_data.sh

Критичное — уже сделано. Что стоит подтянуть вторым кругом:

- Сохранять applied `spark.conf` и tail `bench.log` рядом с каждой строкой
  датасета (не только hash) — поможет при разборе аномалий;
- Логировать `error_type` для failed runs: OOM / timeout / java exception / etc.;
- Добавить `input_records` для workloads, где это можно вытащить
  (KMeans: знаем samples; PageRank: знаем `pagerank.pages`);
- Latin Hypercube / stratified sampling вместо чистого random pick —
  даст лучшее покрытие per cell при том же TARGET_SAMPLES;
- Записывать seed RNG в CSV для воспроизводимости.

## 7. Что ещё сломано или подозрительно

- `models/data.py` и `training/*.py` ссылаются в docstring на `wc_train_all.csv`
  (этот файл перемещён в `data/archive/`). На работу это не влияет, но примеры
  в шапке стоит обновить под `wc_train_merged.csv` / будущий
  `hibench_train_<timestamp>.csv` при следующем коммите в обучалку.
- `docker-compose.yml` объявляет `nodemanager-1` и `nodemanager-2`. `docker
  compose up -d` без аргументов поднимает их автоматически; ранбук раньше
  перечислял сервисы вручную и пропускал NM-ы — эту подсказку надо чинить
  в runbook (см. `docs/VM_DATASET_COLLECTION_RUNBOOK.md`).
- `run_specific_config.sh` в архиве пока WordCount-only; на этапе валидации
  рекомендаций надо переписать его под `collect_hibench_data.sh` и сделать
  generic по `job_type`.
