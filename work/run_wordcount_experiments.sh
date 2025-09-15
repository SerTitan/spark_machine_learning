#!/usr/bin/env bash
set -euo pipefail

# ---- Профили для запуска (huge -> large)
PROFILES=(${PROFILES:-huge large})

# ---- Топология кластера по умолчанию под твою машину (12 vCPU / 27 GiB)
NUM_WORKERS="${NUM_WORKERS:-4}"
WORKER_CORES="${WORKER_CORES:-2}"
WORKER_MEM_GB="${WORKER_MEM_GB:-4}"

# ---- Сетевой бридж из твоего docker-compose
DOCKER_NET="${DOCKER_NET:-spark_machine_learning_bench-net}"

CSV="/opt/hibench/report/wc_train_all.csv"

echo ">>> Останавливаю возможные старые воркеры..."
for i in {1..8}; do docker rm -f "spark-worker-$i" >/dev/null 2>&1 || true; done

echo ">>> Поднимаю ${NUM_WORKERS} воркеров (${WORKER_CORES} cores, ${WORKER_MEM_GB}G mem) в сети ${DOCKER_NET}..."
for i in $(seq 1 "$NUM_WORKERS"); do
  docker run -d --rm \
    --name "spark-worker-$i" \
    --hostname "spark-worker-$i" \
    --network "${DOCKER_NET}" \
    -e SPARK_MODE=worker \
    -e SPARK_MASTER_URL=spark://spark-master:7077 \
    -e SPARK_WORKER_CORES="${WORKER_CORES}" \
    -e SPARK_WORKER_MEMORY="${WORKER_MEM_GB}g" \
    bitnami/spark:3.3 >/dev/null
done

echo ">>> Жду регистрацию воркеров у master..."
# короткий health-wait: проверяем, что порт RPC мастера доступен и даём воркерам «встать»
for s in {1..12}; do
  if docker exec -it hibench bash -lc 'exec 3<>/dev/tcp/spark-master/7077' >/dev/null 2>&1; then
    sleep 4
    break
  fi
  sleep 2
done

echo ">>> Чищу HDFS и CSV..."
docker exec -it hibench bash -lc '
  /opt/hadoop/bin/hdfs dfs -rm -r -skipTrash /Wordcount >/dev/null 2>&1 || true
  mkdir -p /opt/hibench/report
  : > /opt/hibench/report/wc_train_all.csv
'

# ---- Ограничения на число прогонов (можно переопределить переменными окружения)
# huge — компактнее, large — шире
MAX_RUNS_HUGE="${MAX_RUNS_HUGE:-80}"
MAX_RUNS_LARGE="${MAX_RUNS_LARGE:-120}"

for profile in "${PROFILES[@]}"; do
  case "$profile" in
    huge)  max_runs="$MAX_RUNS_HUGE" ;;
    large) max_runs="$MAX_RUNS_LARGE" ;;
    *)     max_runs="${MAX_RUNS_LARGE}" ;;
  esac

  echo ">>> Запускаю сбор ($profile), MAX_RUNS=$max_runs ..."
  docker exec -it \
    -e PROFILE="$profile" \
    -e NUM_WORKERS="$NUM_WORKERS" \
    -e TOTAL_CORES="$((NUM_WORKERS*WORKER_CORES))" \
    -e MAX_RUNS="$max_runs" \
    hibench bash -lc 'bash /opt/hibench/report/collect_wordcount_data.sh'
done

echo ">>> Готово. Краткая сводка по CSV:"
docker exec -it hibench bash -lc '
  wc -l /opt/hibench/report/wc_train_all.csv
  head -n 5 /opt/hibench/report/wc_train_all.csv
'

echo ">>> Останавливаю воркеров..."
for i in $(seq 1 "$NUM_WORKERS"); do docker rm -f "spark-worker-$i" >/dev/null 2>&1 || true; done

echo "ALL DONE ✅"
