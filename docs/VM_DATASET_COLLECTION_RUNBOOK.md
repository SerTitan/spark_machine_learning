# VM Dataset Collection Runbook

Дата: 2026-04-24.

Цель документа: пошагово описать путь от создания облачной VM до smoke test HiBench нагрузок и подготовки к полному сбору датасета.

## 1. Что создаем

Создаем одну VM.

Рекомендуемая конфигурация:

- CPU: 64 vCPU;
- RAM: 128 GB;
- Disk: 600 GB SSD/NVMe;
- OS: Ubuntu 22.04 LTS или Ubuntu 24.04 LTS;
- сеть: обычная, без специальных требований;
- доступ: SSH.

Почему 64 vCPU:

- целевой диапазон Spark параметров включает `executor_cores=1-8` и `executor_instances=1-8`;
- верхняя точка `8 executors x 8 cores` требует 64 executor cores;
- если взять 32 vCPU, верхнюю границу придется ограничивать генератором конфигураций.

Почему 128 GB RAM:

- верхняя точка `8 executors x 8 GB` требует 64 GB только под executors;
- нужны driver, Spark master/history, HDFS/YARN/HiBench/MLflow, OS и файловый cache;
- PageRank и Sort могут активно использовать память и shuffle.

Почему 600 GB disk:

- Docker images;
- Docker volumes HDFS/Spark history/MLflow;
- HiBench input/output;
- Spark event logs;
- raw/clean datasets;
- запас на долгий сбор без аварии по диску.

Текущая VM в Yandex Cloud с `standard-v4a`, `64 cores`, `core_fraction=100`, `128 GB RAM`, `600 GB network-ssd`, `preemptible=false`, `gpus=0` соответствует плану.

## 2. Первичная установка на VM

Подключиться по SSH:

```bash
ssh <user>@<vm-ip>
```

Обновить систему:

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

Поставить базовые утилиты:

```bash
sudo apt-get install -y ca-certificates curl gnupg git jq htop tmux unzip python3 python3-venv python3-pip
```

Установить Docker Engine и Compose plugin:

```bash
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Добавить пользователя в docker group:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

Проверить:

```bash
docker --version
docker compose version
docker run --rm hello-world
```

Если `docker --version` и `docker compose version` работают, но `docker run` или `docker info` говорят `Docker daemon is not reachable`, выполнить:

```bash
sudo systemctl enable --now docker
sudo docker info
sudo usermod -aG docker "$USER"
newgrp docker
docker info
```

Если после `newgrp docker` доступа все еще нет, перелогиниться по SSH.

## 3. Подготовка проекта

Склонировать репозиторий или перенести текущую папку проекта на VM:

```bash
git clone <repo-url> spark_machine_learning_new
cd spark_machine_learning_new
```

Если проект переносится архивом:

```bash
unzip spark_machine_learning_new.zip -d spark_machine_learning_new
cd spark_machine_learning_new
```

Сделать скрипты исполняемыми:

```bash
chmod +x scripts/*.sh
```

Проверить VM и проект:

```bash
MIN_VCPU=32 MIN_MEM_GB=96 MIN_DISK_FREE_GB=250 ./scripts/vm_preflight.sh
```

Для основной VM 64/128/500 можно строже:

```bash
MIN_VCPU=64 MIN_MEM_GB=120 MIN_DISK_FREE_GB=350 ./scripts/vm_preflight.sh
```

## 4. Запуск базового окружения

Поднять все сервисы (HDFS, YARN NM-ы, Spark, HiBench, MLflow):

```bash
docker compose up -d
```

Важно: YARN NodeManager-ы (`nodemanager-1`, `nodemanager-2`) обязательны.
PageRank/KMeans/TeraSort генерируют входные данные через MapReduce, и без NM-ов
prepare висит в состоянии `ACCEPTED`.

Проверить контейнеры:

```bash
docker ps
docker compose ps
```

Проверить HDFS и HiBench:

```bash
docker exec hibench bash -lc '/opt/hadoop/bin/hdfs dfs -ls /'
docker exec hibench bash -lc 'ls -la /opt/hibench/bin/workloads'
docker exec hibench bash -lc 'find /opt/hibench/bin/workloads -maxdepth 3 -type f -name run.sh | sort'
docker exec hibench bash -lc 'find /opt/hibench/bin/workloads -maxdepth 3 -type f -name prepare.sh | sort'
```

## 5. Smoke test трех нагрузок

Запустить smoke test:

```bash
WORKLOADS=kmeans,terasort,wordcount PROFILES=small,large,huge REPEATS=1 ./scripts/smoke_hibench_workloads.sh
```

Если `terasort` в текущем HiBench image не найдется или будет слишком тяжелым, проверяем fallback `sort`:

```bash
WORKLOADS=kmeans,sort,wordcount PROFILES=small,large,huge REPEATS=1 ./scripts/smoke_hibench_workloads.sh
```

Результат появится в:

```text
data/smoke_hibench_workloads_<timestamp>.csv
```

Проверить CSV:

```bash
column -s, -t < data/smoke_hibench_workloads_<timestamp>.csv | less -S
```

Критерии успеха:

- `prepare_rc=0`;
- `run_rc=0`;
- `report_lines_after > report_lines_before`;
- duration выглядит разумно;
- все три нагрузки реально найдены в HiBench image.

Smoke test не собирает обучающий датасет. Его задача: проверить доступность workload paths, корректность `prepare/run` и грубо оценить runtime.

## 6. Как оценить время полного сбора

Точное время нельзя честно назвать до smoke test. Его нужно считать из фактических `duration_s`.

Формула:

```text
total_time_seconds =
  workloads
  x profiles
  x topologies
  x configs_per_cell
  x repeats
  x avg_run_seconds
  x overhead_factor
```

Где:

- `workloads = 3`;
- `profiles = 3`: small, large, huge для smoke-проверки текущего HiBench image;
- `topologies = 3 или 4`;
- `repeats = 5` для финального датасета;
- `overhead_factor = 1.2-1.4`.

Безопасный стартовый план на одну VM:

```text
3 workloads x 3 profiles x 3 topologies x 20 configs x 5 repeats
= 2700 Spark runs
```

Если средний run после smoke test:

- 20 sec: примерно 18 часов с overhead;
- 30 sec: примерно 30 часов с overhead;
- 45 sec: примерно 43 часа с overhead;
- 60 sec: примерно 57 часов с overhead.

Если хотим гарантированно остаться ближе к 2 суткам, правило такое:

- avg <= 45 sec: можно 20 configs/cell;
- avg 45-70 sec: лучше 12-15 configs/cell;
- avg > 70 sec: heavy workload собирать отдельно с меньшим числом configs или 3 repeats.

Для одной нагрузки, например KMeans:

```text
1 workload x 3 profiles x 3 topologies x 40 configs x 5 repeats
= 1800 Spark runs
```

Оценка:

- 30 sec/run: примерно 18-21 часов;
- 45 sec/run: примерно 30-34 часа;
- 60 sec/run: примерно 40-45 часов.

Поэтому реалистичный порядок:

1. Smoke test всех трех нагрузок.
2. Первый полноценный сбор KMeans в ограниченной матрице, если smoke test пройдет.
3. Обучение на первом датасете.
4. Расширение TeraSort/WordCount после проверки фактических времен.

## 7. Последовательный сбор

На одной VM сбор запускаем последовательно. Параллельные Spark workload на той же машине запрещены для чистого датасета: они конкурируют за CPU/RAM/disk/network.

Рекомендуемая очередность:

1. KMeans: первая новая ключевая нагрузка, если починили генератор входных данных.
2. TeraSort: shuffle-heavy нагрузка, если smoke test пройдет.
3. WordCount: контрольная нагрузка и сравнение с прошлым датасетом.

Если TeraSort не пройдет smoke test, заменяем его на Sort и фиксируем причину в отчете.

Универсальные скрипты уже добавлены:

- `scripts/run_hibench_experiments.sh` запускается на host VM;
- `scripts/collect_hibench_data.sh` запускается внутри контейнера `hibench`;
- `scripts/smoke_hibench_workloads.sh` проверяет доступность нагрузок;
- `scripts/snapshot_hdfs_input.sh` сохраняет HDFS input в архив;
- `scripts/restore_hdfs_input.sh` восстанавливает HDFS input.

Новый collector должен принимать:

```bash
JOB_TYPE=kmeans|terasort|wordcount
PROFILE=small|large|huge
WORKLOAD_DIR=/opt/hibench/bin/workloads/...
TARGET_SAMPLES=...
REPEATS=...
NUM_WORKERS=...
WORKER_CORES=...
WORKER_MEM_GB=...
```

Текущий runner использует 3 топологии:

```text
8 workers x 8 cores x 8 GB
4 workers x 8 cores x 16 GB
8 workers x 4 cores x 8 GB
```

Они дают покрытие диапазона `executor_cores=1-8`, `executor_instances=1-8`, `executor_memory=1-8g`, но runner дополнительно ограничивает опасные сочетания:

```text
MAX_TOTAL_EXECUTOR_CORES=56
MAX_TOTAL_EXECUTOR_MEMORY_GB=90
```

Это оставляет запас под OS, Docker, HDFS, Spark master/history, HiBench и driver. Значения можно переопределить через переменные окружения, но для основного сбора лучше оставить defaults.

Запуск полного сбора:

```bash
WORKLOADS=kmeans,terasort,wordcount \
PROFILES=small,large,huge \
TARGET_SAMPLES=20 \
REPEATS=5 \
./scripts/run_hibench_experiments.sh
```

Если TeraSort заменяем на Sort:

```bash
WORKLOADS=kmeans,sort,wordcount \
PROFILES=small,large,huge \
TARGET_SAMPLES=20 \
REPEATS=5 \
./scripts/run_hibench_experiments.sh
```

Collector пишет расширенную CSV-схему:

```text
schema_version,experiment_id,timestamp,job_type,profile,input_dataset_id,input_hdfs_path,input_size_bytes,
topology_workers,topology_worker_cores,topology_worker_mem_gb,
total_cores,total_memory_gb,
executor_cores,executor_memory,executor_instances,
driver_cores,driver_memory,
memory_fraction,memory_storageFraction,
shuffle_compress,spill_compress,shuffle_file_buffer,
broadcast_block,broadcast_compress,maxSizeInFlight,
io_codec,rpc_message_maxSize,rdd_compress,
run_durations_s,
median_duration_s,mean_duration_s,std_duration_s,min_duration_s,max_duration_s,cv_duration,
successful_runs,exit_code,spark_conf_hash
```

## 8. Контроль диска и Docker cache

Перед долгим сбором открыть tmux:

```bash
tmux new -s spark-collect
```

В первом окне запустить мониторинг:

```bash
./scripts/monitor_collection_resources.sh
```

По умолчанию он пишет snapshot каждые 300 секунд в:

```text
logs/collection_resources_<timestamp>.log
```

Можно чаще:

```bash
INTERVAL=120 ./scripts/monitor_collection_resources.sh
```

Ручные команды контроля:

```bash
df -h
docker system df
docker ps -a
docker volume ls
du -sh data out logs
docker exec hibench bash -lc '/opt/hadoop/bin/hdfs dfs -du -h / | sort -h'
```

Безопасная чистка во время паузы между батчами:

```bash
docker container prune -f
docker image prune -f
docker builder prune -f
```

Что нельзя делать во время активного сбора:

```bash
docker system prune -af --volumes
```

Эта команда может удалить volumes с HDFS/Spark history/MLflow и сломать экспериментальные данные.

После каждого батча нужно копировать CSV на host и делать snapshot:

```bash
mkdir -p data/snapshots
cp data/<dataset>.csv data/snapshots/<dataset>_$(date +%Y%m%d_%H%M%S).csv
```

## 9. Вход сервиса

Внешний пользователь не вводит `small/medium/large`. Он вводит размер данных и доступные ресурсы.

Минимальный запрос:

```json
{
  "job_type": "pagerank",
  "input_size_gb": 7,
  "resources": {
    "total_cores": 64,
    "total_memory_gb": 128,
    "disk_available_gb": 500
  }
}
```

Расширенный запрос:

```json
{
  "job_type": "pagerank",
  "input_size_gb": 7,
  "resources": {
    "total_cores": 64,
    "total_memory_gb": 128,
    "disk_available_gb": 500,
    "nodes": 1
  },
  "constraints": {
    "max_executor_cores": 8,
    "max_executor_memory_gb": 8,
    "max_executor_instances": 8
  },
  "preferences": {
    "return_top_k": 3,
    "optimize_for": "runtime"
  }
}
```

Внутри модели полезно хранить и агрегированные ресурсы, и форму кластера:

- `total_cores`;
- `total_memory_gb`;
- `disk_available_gb`;
- `nodes/workers`;
- `worker_cores`;
- `worker_memory_gb`;
- `input_size_bytes`;
- `job_type`.

Spark-параметры сервис должен рекомендовать сам.

## 10. Вопрос про разный датасет внутри одной нагрузки

Да, это реальный риск. Если каждый `prepare` генерирует другой набор данных, время может зависеть не только от Spark config, но и от конкретного содержимого input.

Правильная защита:

1. Для одной ячейки `job_type + profile/input_size` готовить input один раз.
2. Не перегенерировать input перед каждой Spark config.
3. Все конфигурации внутри этой ячейки гонять на одном и том же HDFS input.
4. В датасет записывать `input_dataset_id` и `input_size_bytes`.
5. Если хотим оценить влияние разных input seeds, делать это отдельным controlled experiment:
   - `input_seed=1`, `input_seed=2`, `input_seed=3`;
   - одинаковый набор Spark configs;
   - потом оценить разброс.

Для финального основного сбора лучше:

- один prepared input на `job_type/profile`;
- несколько повторов run для каждой Spark config;
- target = median по повторам.

Для проверки устойчивости можно добавить малый validation-блок:

```text
3 workloads x 3 profiles x 2 input seeds x 5 configs x 5 repeats
```

Это покажет, насколько модель чувствительна к конкретному содержимому данных.

## 10.1. Snapshot HDFS input между нагрузками

Чтобы не потерять входные данные после `prepare`, добавлены host-side скрипты:

- `scripts/snapshot_hdfs_input.sh`;
- `scripts/restore_hdfs_input.sh`.

Порядок для каждой пары `job_type + profile`:

1. HiBench делает `prepare` и создает input в HDFS.
2. Мы определяем HDFS path input.
3. Делаем snapshot:

```bash
JOB_TYPE=pagerank \
PROFILE=large \
INPUT_DATASET_ID=pagerank_large_001 \
HDFS_PATH=/path/to/hibench/input \
./scripts/snapshot_hdfs_input.sh
```

4. Скрипт:
   - проверяет, что `HDFS_PATH` существует;
   - делает `hdfs dfs -get` во временную папку контейнера `hibench`;
   - сжимает input в `tar.gz`;
   - копирует архив на host в `data/hdfs_inputs/<job_type>/<profile>/`;
   - пишет metadata JSON рядом с архивом.

5. После snapshot можно запускать сбор конфигураций на этом input.
6. Перед переходом к следующей нагрузке snapshot уже лежит на host-диске VM, поэтому input можно восстановить позже.

Восстановление:

```bash
ARCHIVE_PATH=data/hdfs_inputs/pagerank/large/pagerank_large_001.tar.gz \
TARGET_HDFS_PATH=/path/to/hibench/input \
./scripts/restore_hdfs_input.sh
```

Что сохраняется:

```text
data/hdfs_inputs/
  pagerank/
    large/
      pagerank_large_001.tar.gz
      pagerank_large_001.metadata.json
```

Отдельный HDFS-контейнер на локальной машине не обязателен. Достаточно хранить сжатые snapshot-архивы на host-диске VM и периодически скачивать их с VM через `scp`/`rsync`.

Пример скачать с VM на локальную машину:

```bash
rsync -avz sertitan@<vm-ip>:~/spark_machine_learning_new/data/hdfs_inputs/ ./data/hdfs_inputs/
```

## 11. Ближайшие действия

1. Создать VM `64 vCPU / 128 GB RAM / 600 GB SSD`.
2. Установить Docker Engine и Compose plugin.
3. Перенести проект.
4. Запустить `scripts/vm_preflight.sh`.
5. Поднять compose.
6. Запустить `scripts/smoke_hibench_workloads.sh`.
7. По smoke CSV посчитать реальные времена.
8. Зафиксировать финальную матрицу сбора.
9. Запустить первый полноценный сбор PageRank.
10. Затем TeraSort или Sort.
11. Затем WordCount.
12. После каждого батча проверять CSV, snapshot и состояние диска.
13. После первого батча начать обучение моделей, если это не мешает активному сбору на VM.
