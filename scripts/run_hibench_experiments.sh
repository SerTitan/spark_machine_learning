#!/usr/bin/env bash
set -euo pipefail
#
# run_hibench_experiments.sh — host-runner полного сбора датасета.
#
# Запускается на хосте (Yandex Cloud VM). Обходит матрицу
# WORKLOADS × PROFILES × TOPOLOGIES, для каждой ячейки:
#   1) поднимает нужное число Spark-воркеров с заданными CPU/RAM;
#   2) копирует generic-коллектор в контейнер hibench;
#   3) запускает collect_hibench_data.sh внутри контейнера;
#   4) забирает CSV-строки на хост и приклеивает их к итоговому датасету.
#
# Топологии (6 штук) специально подобраны так, чтобы покрыть основные
# режимы исполнения Spark в пределах 64 vCPU:
#   - 2×4×8, 4×4×8 — малые baseline-кластера;
#   - 4×8×16 vs 8×4×8 — два варианта с одинаковыми Σcores=32, разной формы;
#   - 6×6×12, 8×6×12 — потолок в рамках 64 vCPU (с запасом под NM/HDFS).
# Это даёт модели возможность отличить эффект "много узких" от "мало широких"
# исполнителей при тех же суммарных ресурсах.
#
# Пример: WORKLOADS=terasort PROFILES=small,large TARGET_SAMPLES=80 REPEATS=3 \
#         OUT_CSV=data/hibench_ts_$(date +%Y%m%d).csv ./scripts/run_hibench_experiments.sh
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
DATA_DIR="${PROJECT_DIR}/data"
SNAPSHOTS_DIR="${DATA_DIR}/snapshots"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

DOCKER_NET="${DOCKER_NET:-spark_machine_learning_bench-net}"
SPARK_WORKER_IMAGE="${SPARK_WORKER_IMAGE:-sertitanius/spark_machine_learning-spark:3.3}"
WORKLOADS="${WORKLOADS:-terasort}"
PROFILES="${PROFILES:-small,large}"
TARGET_SAMPLES="${TARGET_SAMPLES:-80}"
REPEATS="${REPEATS:-3}"
MAX_TOTAL_EXECUTOR_CORES="${MAX_TOTAL_EXECUTOR_CORES:-48}"
MAX_TOTAL_EXECUTOR_MEMORY_GB="${MAX_TOTAL_EXECUTOR_MEMORY_GB:-90}"
OUT_CSV="${OUT_CSV:-${DATA_DIR}/hibench_train_${TIMESTAMP}.csv}"
CONTAINER_CSV="${CONTAINER_CSV:-/opt/hibench/report/hibench_train_all.csv}"
WORKER_CONTAINER_CPUS="${WORKER_CONTAINER_CPUS:-auto}"
WORKER_CONTAINER_MEM_GB="${WORKER_CONTAINER_MEM_GB:-auto}"
RNG_SEED="${RNG_SEED:-20260424}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${DATA_DIR}/collection_artifacts_${TIMESTAMP}}"
WORKLOAD_EXPORT_DIR="${WORKLOAD_EXPORT_DIR:-${DATA_DIR}/workload_exports_${TIMESTAMP}}"
SNAPSHOT_HDFS_INPUTS="${SNAPSHOT_HDFS_INPUTS:-1}"
CLEAN_HDFS_INPUTS_AFTER_SNAPSHOT="${CLEAN_HDFS_INPUTS_AFTER_SNAPSHOT:-1}"
SAVE_SPARK_CONF_ARTIFACTS="${SAVE_SPARK_CONF_ARTIFACTS:-0}"
SAVE_BENCH_LOG_TAILS="${SAVE_BENCH_LOG_TAILS:-0}"

mkdir -p "$DATA_DIR" "$SNAPSHOTS_DIR" "$ARTIFACTS_DIR" "$WORKLOAD_EXPORT_DIR"

TOPOLOGIES=(
  "workers=2;cores=4;mem=8"
  "workers=4;cores=4;mem=8"
  "workers=4;cores=8;mem=16"
  "workers=8;cores=4;mem=8"
  "workers=6;cores=6;mem=12"
  "workers=8;cores=6;mem=12"
)

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $1" >&2
    exit 1
  fi
}

need_cmd docker

# Preflight: YARN NodeManagers must be up for PageRank/KMeans/TeraSort prepare.
NM_COUNT="$(docker exec resourcemanager bash -lc "/opt/hadoop/bin/yarn node -list 2>/dev/null | grep -c RUNNING" 2>/dev/null || echo 0)"
if [[ "${NM_COUNT}" -lt 1 ]]; then
  echo "ERROR: no YARN NodeManagers registered. Start them with:" >&2
  echo "  docker compose up -d nodemanager-1 nodemanager-2" >&2
  echo "Without NMs, PageRank/KMeans/TeraSort prepare will hang." >&2
  exit 1
fi
echo ">>> YARN NodeManagers online: ${NM_COUNT}"

kill_workers() {
  for i in $(seq 1 32); do
    docker rm -f "spark-worker-$i" >/dev/null 2>&1 || true
  done
}

start_workers() {
  local n="$1" c="$2" m="$3"
  local container_cpus="$c"
  local container_mem_gb="$m"
  if [[ "$WORKER_CONTAINER_CPUS" != "auto" ]]; then
    container_cpus="$WORKER_CONTAINER_CPUS"
  fi
  if [[ "$WORKER_CONTAINER_MEM_GB" != "auto" ]]; then
    container_mem_gb="$WORKER_CONTAINER_MEM_GB"
  fi
  echo ">>> Starting Spark workers: ${n} workers x ${c} cores x ${m} GB"
  kill_workers
  for i in $(seq 1 "$n"); do
    docker run -d --rm \
      --name "spark-worker-$i" \
      --hostname "spark-worker-$i" \
      --network "$DOCKER_NET" \
      --cpus "$container_cpus" \
      --memory "${container_mem_gb}g" \
      --memory-swap "${container_mem_gb}g" \
      -e SPARK_MODE=worker \
      -e SPARK_MASTER_URL=spark://spark-master:7077 \
      -e SPARK_WORKER_CORES="$c" \
      -e SPARK_WORKER_MEMORY="${m}g" \
      "$SPARK_WORKER_IMAGE" >/dev/null
  done
  sleep 8
}

resolve_workload_dir() {
  local workload="$1"
  docker exec hibench bash -lc "python3 - <<'PY'
from pathlib import Path
import sys

name = '${workload}'.lower()
candidates = {
    'wordcount': [
        '/opt/hibench/bin/workloads/micro/wordcount',
    ],
    'pagerank': [
        '/opt/hibench/bin/workloads/websearch/pagerank',
        '/opt/hibench/bin/workloads/graph/pagerank',
    ],
    'kmeans': [
        '/opt/hibench/bin/workloads/ml/kmeans',
        '/opt/hibench/bin/workloads/machinelearning/kmeans',
    ],
    'sort': [
        '/opt/hibench/bin/workloads/micro/sort',
    ],
    'terasort': [
        '/opt/hibench/bin/workloads/micro/terasort',
    ],
}.get(name, [])

for raw in candidates:
    p = Path(raw)
    if (p / 'prepare' / 'prepare.sh').exists() and (p / 'spark' / 'run.sh').exists():
        print(p)
        sys.exit(0)

for p in Path('/opt/hibench/bin/workloads').rglob(name):
    if p.is_dir() and (p / 'prepare' / 'prepare.sh').exists() and (p / 'spark' / 'run.sh').exists():
        print(p)
        sys.exit(0)

sys.exit(1)
PY"
}

expected_input_path() {
  local workload="$1"
  case "${workload,,}" in
    wordcount) echo "/Wordcount/Input" ;;
    pagerank) echo "/Pagerank/Input" ;;
    kmeans) echo "/Kmeans/Input" ;;
    sort|terasort) echo "/Terasort/Input" ;;
    *) echo "unknown" ;;
  esac
}

export_workload_csv() {
  local workload="$1"
  local src="$2"
  local dst="${WORKLOAD_EXPORT_DIR}/${workload}_$(basename "$src")"
  if [[ -s "$src" ]]; then
    awk -F, -v w="$workload" 'NR == 1 || $4 == w { print }' "$src" > "$dst"
    echo ">>> Workload export saved: ${dst} rows=$(tail -n +2 "$dst" | wc -l)"
  fi
}

snapshot_hdfs_input() {
  local workload="$1" profile="$2" input_dataset_id="$3" hdfs_path="$4"
  if [[ "$SNAPSHOT_HDFS_INPUTS" != "1" || "$hdfs_path" == "unknown" ]]; then
    return 0
  fi
  if ! docker exec hibench bash -lc "/opt/hadoop/bin/hdfs dfs -test -e '${hdfs_path}'" >/dev/null 2>&1; then
    echo "WARN: HDFS input does not exist, snapshot skipped: ${hdfs_path}"
    return 0
  fi

  echo ">>> Snapshotting HDFS input: workload=${workload}; profile=${profile}; path=${hdfs_path}"
  JOB_TYPE="$workload" \
    PROFILE="$profile" \
    INPUT_DATASET_ID="$input_dataset_id" \
    HDFS_PATH="$hdfs_path" \
    "${SCRIPT_DIR}/snapshot_hdfs_input.sh"

  if [[ "$CLEAN_HDFS_INPUTS_AFTER_SNAPSHOT" == "1" ]]; then
    echo ">>> Removing uncompressed HDFS input after snapshot: ${hdfs_path}"
    docker exec hibench bash -lc "/opt/hadoop/bin/hdfs dfs -rm -r -f -skipTrash '${hdfs_path}'"
  fi
}

echo "=== Generic HiBench dataset collection ==="
echo "Workloads: ${WORKLOADS}"
echo "Profiles: ${PROFILES}"
echo "Target samples per cell: ${TARGET_SAMPLES}"
echo "Repeats: ${REPEATS}"
echo "RNG seed: ${RNG_SEED}"
echo "Output CSV: ${OUT_CSV}"
echo "Artifacts dir: ${ARTIFACTS_DIR}"
echo "Workload exports dir: ${WORKLOAD_EXPORT_DIR}"
echo "Snapshot HDFS inputs: ${SNAPSHOT_HDFS_INPUTS}"
echo "Clean HDFS inputs after snapshot: ${CLEAN_HDFS_INPUTS_AFTER_SNAPSHOT}"
echo "Save per-run spark.conf artifacts: ${SAVE_SPARK_CONF_ARTIFACTS}"
echo "Save per-run bench.log tails: ${SAVE_BENCH_LOG_TAILS}"

docker exec hibench bash -lc 'mkdir -p /opt/hibench/report && rm -f /opt/hibench/report/hibench_train_all.csv'
docker cp "${SCRIPT_DIR}/collect_hibench_data.sh" hibench:/opt/hibench/report/collect_hibench_data.sh
docker exec hibench bash -lc 'chmod +x /opt/hibench/report/collect_hibench_data.sh'

IFS=',' read -r -a workload_arr <<< "$WORKLOADS"
IFS=',' read -r -a profile_arr <<< "$PROFILES"

for workload in "${workload_arr[@]}"; do
  workload="$(echo "$workload" | xargs)"
  echo "=== Workload: ${workload} ==="

  set +e
  workload_dir="$(resolve_workload_dir "$workload")"
  rc=$?
  set -e
  if [[ "$rc" -ne 0 || -z "$workload_dir" ]]; then
    echo "ERROR: workload not found or unsupported in this HiBench image: ${workload}" >&2
    exit 1
  fi
  echo ">>> Resolved path: ${workload_dir}"

  for profile in "${profile_arr[@]}"; do
    profile="$(echo "$profile" | xargs)"
    input_dataset_id="${workload}_${profile}_${TIMESTAMP}"
    input_hdfs_path="$(expected_input_path "$workload")"

    for idx in "${!TOPOLOGIES[@]}"; do
      eval "${TOPOLOGIES[$idx]}"
      echo "=== Cell: workload=${workload}; profile=${profile}; topology=${workers}x${cores}x${mem} ==="
      start_workers "$workers" "$cores" "$mem"

      prepare_input="0"
      if [[ "$idx" == "0" ]]; then
        prepare_input="1"
      fi
      docker exec \
        -e JOB_TYPE="$workload" \
        -e WORKLOAD_DIR="$workload_dir" \
        -e PROFILE="$profile" \
        -e NUM_WORKERS="$workers" \
        -e WORKER_CORES="$cores" \
        -e WORKER_MEM_GB="$mem" \
        -e TARGET_SAMPLES="$TARGET_SAMPLES" \
        -e REPEATS="$REPEATS" \
        -e PREPARE_INPUT="$prepare_input" \
        -e INPUT_DATASET_ID="$input_dataset_id" \
        -e INPUT_HDFS_PATH="$input_hdfs_path" \
        -e INPUT_SIZE_BYTES="auto" \
        -e MAX_TOTAL_EXECUTOR_CORES="$MAX_TOTAL_EXECUTOR_CORES" \
        -e MAX_TOTAL_EXECUTOR_MEMORY_GB="$MAX_TOTAL_EXECUTOR_MEMORY_GB" \
        -e RNG_SEED="$RNG_SEED" \
        -e SAVE_SPARK_CONF_ARTIFACTS="$SAVE_SPARK_CONF_ARTIFACTS" \
        -e SAVE_BENCH_LOG_TAILS="$SAVE_BENCH_LOG_TAILS" \
        -e CSV="$CONTAINER_CSV" \
        hibench bash -lc '/opt/hibench/report/collect_hibench_data.sh'

      docker cp "hibench:${CONTAINER_CSV}" "$OUT_CSV"
      docker cp "hibench:/opt/hibench/report/collection_artifacts" "$ARTIFACTS_DIR" >/dev/null 2>&1 || true
      cp "$OUT_CSV" "${SNAPSHOTS_DIR}/$(basename "$OUT_CSV" .csv)_snapshot.csv"
      export_workload_csv "$workload" "$OUT_CSV"
      echo ">>> Snapshot saved. Current rows: $(tail -n +2 "$OUT_CSV" | wc -l)"
    done
    snapshot_hdfs_input "$workload" "$profile" "$input_dataset_id" "$input_hdfs_path"
  done
  export_workload_csv "$workload" "$OUT_CSV"
done

kill_workers
docker cp "hibench:${CONTAINER_CSV}" "$OUT_CSV"
docker cp "hibench:/opt/hibench/report/collection_artifacts" "$ARTIFACTS_DIR" >/dev/null 2>&1 || true
for workload in "${workload_arr[@]}"; do
  workload="$(echo "$workload" | xargs)"
  export_workload_csv "$workload" "$OUT_CSV"
done
echo ">>> Final dataset saved to: ${OUT_CSV}"
echo ">>> Collection artifacts copied to: ${ARTIFACTS_DIR}"
echo ">>> Total samples: $(tail -n +2 "$OUT_CSV" | wc -l)"
