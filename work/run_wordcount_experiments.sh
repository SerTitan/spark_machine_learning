#!/usr/bin/env bash
set -euo pipefail
#
# run_wordcount_experiments.sh — Запускается на ХОСТЕ.
#

DOCKER_NET="${DOCKER_NET:-spark_machine_learning_bench-net}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-500}"
REPEATS="${REPEATS:-9}"
PROFILE="large"

TOPOLOGIES=(
  "workers=2;cores=4;mem=8"
  "workers=3;cores=3;mem=6"
  "workers=4;cores=2;mem=4"
)

per=$(( TOTAL_SAMPLES / 3 ))
rem=$(( TOTAL_SAMPLES % 3 ))
TARGETS=( $((per+rem)) $per $per )

echo "=== SPARK WordCount dataset collection ==="

_kill_workers() { for i in {1..12}; do docker rm -f "spark-worker-$i" >/dev/null 2>&1 || true; done; }
_start_workers() {
  local n="$1" c="$2" m="$3"
  for i in $(seq 1 "$n"); do
    docker run -d --rm --name "spark-worker-$i" --hostname "spark-worker-$i" --network "${DOCKER_NET}" \
      -e SPARK_MODE=worker -e SPARK_MASTER_URL=spark://spark-master:7077 \
      -e SPARK_WORKER_CORES="${c}" -e SPARK_WORKER_MEMORY="${m}g" bitnami/spark:3.3 >/dev/null
  done
  for s in {1..20}; do if (exec 3<>/dev/tcp/localhost/7077) >/dev/null 2>&1; then sleep 4; break; fi; sleep 1; done
}

# Подготовка данных (однократно!)
echo ">>> Prepare Wordcount input (profile=$PROFILE)..."
docker exec -it hibench bash -lc "sed -i 's/^hibench.scale.profile.*/hibench.scale.profile        ${PROFILE}/' conf/hibench.conf && bin/workloads/micro/wordcount/prepare/prepare.sh || true"

# Готовим скрипт
docker cp "$(dirname "$0")/collect_wordcount_data.sh" hibench:/opt/hibench/report/collect_wordcount_data.sh
docker exec -it hibench bash -lc 'chmod +x /opt/hibench/report/collect_wordcount_data.sh && mkdir -p /opt/hibench/report && : > /opt/hibench/report/wc_train_all.csv'

for idx in "${!TOPOLOGIES[@]}"; do
  eval "${TOPOLOGIES[$idx]}"; target="${TARGETS[$idx]}"
  echo "=== Topology $((idx+1))/3: ${workers}×${cores}c×${mem}g | samples=$target ==="
  _kill_workers; _start_workers "$workers" "$cores" "$mem"
  docker exec -it -e PROFILE="$PROFILE" -e NUM_WORKERS="$workers" -e WORKER_CORES="$cores" -e WORKER_MEM_GB="$mem" -e TARGET_SAMPLES="$target" -e REPEATS="$REPEATS" hibench bash -lc '/opt/hibench/report/collect_wordcount_data.sh'
done

echo ">>> Final CSV preview:"; docker exec -it hibench bash -lc 'wc -l /opt/hibench/report/wc_train_all.csv && tail -n 5 /opt/hibench/report/wc_train_all.csv'
_kill_workers
