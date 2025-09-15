#!/usr/bin/env bash
set -euo pipefail

# --- Профиль данных HiBench (tiny/small/large/huge/custom)
PROFILE="${PROFILE:-huge}"

# --- Топология воркеров (вписывается в 12 vCPU / 27 GiB)
NUM_WORKERS="${NUM_WORKERS:-4}"
WORKER_CORES="${WORKER_CORES:-2}"
WORKER_MEM_GB="${WORKER_MEM_GB:-4}"

# --- Где CSV
CSV="/opt/hibench/report/wc_train_all.csv"

echo ">>> Останавливаю старые воркеры, если есть..."
for i in {1..8}; do docker rm -f "spark-worker-$i" >/dev/null 2>&1 || true; done

echo ">>> Поднимаю ${NUM_WORKERS} воркеров (${WORKER_CORES} cores, ${WORKER_MEM_GB}G memory)..."
for i in $(seq 1 "$NUM_WORKERS"); do
  docker run -d --rm \
    --name "spark-worker-$i" \
    --hostname "spark-worker-$i" \
    --network spark_machine_learning_bench-net \
    -e SPARK_MODE=worker \
    -e SPARK_MASTER_URL=spark://spark-master:7077 \
    -e SPARK_WORKER_CORES="${WORKER_CORES}" \
    -e SPARK_WORKER_MEMORY="${WORKER_MEM_GB}g" \
    bitnami/spark:3.3 >/dev/null
done

echo ">>> Жду регистрации воркеров у мастера..."
# Просто маленький таймер + проверка TCP порта
for s in {1..20}; do
  if docker exec -it hibench bash -lc 'exec 3<>/dev/tcp/spark-master/7077' >/dev/null 2>&1; then
    sleep 4; break
  fi
  sleep 1
done

echo ">>> Чищу HDFS и CSV..."
docker exec -it hibench bash -lc '
  /opt/hadoop/bin/hdfs dfs -rm -r -skipTrash /Wordcount >/dev/null 2>&1 || true
  mkdir -p /opt/hibench/report
  : > /opt/hibench/report/wc_train_all.csv
'

echo ">>> Запускаю сбор обучающей выборки (grid-search)..."
docker exec -it -e PROFILE="${PROFILE}" -e NUM_WORKERS="${NUM_WORKERS}" \
  -e TOTAL_CORES="$((NUM_WORKERS*WORKER_CORES))" \
  -e MAX_RUNS="${MAX_RUNS:-128}" \
  hibench bash -lc 'bash /opt/hibench/report/collect_wordcount_data.sh'

echo ">>> Готово. CSV:"
docker exec -it hibench bash -lc 'wc -l /opt/hibench/report/wc_train_all.csv && head -n 5 /opt/hibench/report/wc_train_all.csv | sed -n "1,5p"'

echo ">>> Останавливаю воркеров..."
for i in $(seq 1 "$NUM_WORKERS"); do docker rm -f "spark-worker-$i" >/dev/null 2>&1 || true; done

echo "ALL DONE ✅"
