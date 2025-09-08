#!/usr/bin/env bash
set -euo pipefail

# ===== Defaults (override by flags) =====
RUNS="${RUNS:-9}"
SCALE="${SCALE:-tiny}"               # tiny/small/large/huge
OUT_DIR="${OUT_DIR:-./out}"
SKIP_PREPARE="${SKIP_PREPARE:-1}"    # 1=skip input prepare
AUTO_PARALLELISM="${AUTO_PARALLELISM:-1}"  # 1=choose parallelism by SCALE if not provided
IS_DEFAULT="${IS_DEFAULT:-0}"

# Spark params
EXECUTORS="${EXECUTORS:-2}"
EXECUTOR_CORES="${EXECUTOR_CORES:-1}"
EXECUTOR_MEMORY_GB="${EXECUTOR_MEMORY_GB:-2}"
DRIVER_CORES="${DRIVER_CORES:-1}"
DRIVER_MEMORY_GB="${DRIVER_MEMORY_GB:-1}"
PARALLELISM="${PARALLELISM:-}"
SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-}"
RDD_COMPRESS="${RDD_COMPRESS:-true}"
IO_COMPRESSION_CODEC="${IO_COMPRESSION_CODEC:-lz4}"
INPUT_GB="${INPUT_GB:-1}"
ENGINE="${ENGINE:-spark}"

# ----- Parse flags -----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs) RUNS="$2"; shift 2;;
    --scale) SCALE="$2"; shift 2;;
    --outdir|--out-dir) OUT_DIR="$2"; shift 2;;
    --skip-prepare) SKIP_PREPARE="$2"; shift 2;;
    --auto-parallelism) AUTO_PARALLELISM="$2"; shift 2;;
    --is-default) IS_DEFAULT="$2"; shift 2;;

    --executors) EXECUTORS="$2"; shift 2;;
    --executor-cores) EXECUTOR_CORES="$2"; shift 2;;
    --executor-memory-gb) EXECUTOR_MEMORY_GB="$2"; shift 2;;
    --driver-cores) DRIVER_CORES="$2"; shift 2;;
    --driver-memory-gb) DRIVER_MEMORY_GB="$2"; shift 2;;
    --parallelism) PARALLELISM="$2"; shift 2;;
    --shuffle-partitions) SHUFFLE_PARTITIONS="$2"; shift 2;;
    --rdd-compress) RDD_COMPRESS="$2"; shift 2;;
    --io-compression-codec) IO_COMPRESSION_CODEC="$2"; shift 2;;
    --input-gb) INPUT_GB="$2"; shift 2;;
    --engine) ENGINE="$2"; shift 2;;

    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

# ----- Auto-parallelism if not provided -----
if [[ -z "${PARALLELISM}" || -z "${SHUFFLE_PARTITIONS}" ]]; then
  if [[ "${AUTO_PARALLELISM}" == "1" ]]; then
    case "${SCALE}" in
      tiny)  DEF_PAR=8 ;;
      small) DEF_PAR=64 ;;
      large) DEF_PAR=256 ;;
      huge)  DEF_PAR=512 ;;
      *)     DEF_PAR=128 ;;
    esac
    PARALLELISM="${PARALLELISM:-$DEF_PAR}"
    SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-$DEF_PAR}"
  else
    PARALLELISM="${PARALLELISM:-128}"
    SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-128}"
  fi
fi

SPARK_CONF="--conf spark.executor.instances=${EXECUTORS} \
--conf spark.executor.cores=${EXECUTOR_CORES} \
--conf spark.executor.memory=${EXECUTOR_MEMORY_GB}g \
--conf spark.driver.cores=${DRIVER_CORES} \
--conf spark.driver.memory=${DRIVER_MEMORY_GB}g \
--conf spark.default.parallelism=${PARALLELISM} \
--conf spark.sql.shuffle.partitions=${SHUFFLE_PARTITIONS} \
--conf spark.rdd.compress=${RDD_COMPRESS} \
--conf spark.io.compression.codec=${IO_COMPRESSION_CODEC}"

#echo "[0] Build & up (hibench)…"
#docker compose build hibench >/dev/null
#docker compose up -d

echo "[1] Health checks…"
./bin/up_all_and_check.sh || true

echo "[1.5] Short settle sleep before submitting jobs…"
sleep 3

# ----- Preflight resources: sum worker cores/mem and compare -----
echo "[1.6] Preflight: checking worker resources vs requested executors…"
WORKERS=$(docker ps --format '{{.Names}}' | grep -E '^spark-worker' || true)
if [[ -z "${WORKERS}" ]]; then
  echo "WARN: No spark-worker containers found; skipping preflight."
else
  SUM_CORES=0
  SUM_MEM_GB=0
  for W in ${WORKERS}; do
    CORES=$(docker exec -i "$W" bash -lc 'echo -n ${SPARK_WORKER_CORES:-0}' 2>/dev/null || echo 0)
    MEM=$(docker exec -i "$W" bash -lc 'echo -n ${SPARK_WORKER_MEMORY:-0}' 2>/dev/null || echo 0)
    MEM_GB=$(echo "$MEM" | awk '{g=toupper($0); sub(/G$/,"",g); sub(/GB$/,"",g); print g}')
    [[ -z "$CORES" ]] && CORES=0
    [[ -z "$MEM_GB" ]] && MEM_GB=0
    SUM_CORES=$((SUM_CORES + CORES))
    SUM_MEM_GB=$((SUM_MEM_GB + MEM_GB))
  done
  REQ_CORES=$((EXECUTORS * EXECUTOR_CORES))
  OVERHEAD=$(python3 - <<PY
m=${EXECUTOR_MEMORY_GB}
print(max(0.384, 0.1*float(m)))
PY
)
  REQ_MEM=$(python3 - <<PY
e=${EXECUTORS}; m=${EXECUTOR_MEMORY_GB}; ov=${OVERHEAD}
print(int(e*(m+ov)))
PY
)
  echo "Workers total: ${SUM_CORES} cores, ${SUM_MEM_GB} GiB; Requested: ${REQ_CORES} cores, ${REQ_MEM} GiB (incl. overhead≈${OVERHEAD}g per executor)"
  if [[ ${REQ_CORES} -gt ${SUM_CORES} || ${REQ_MEM} -gt ${SUM_MEM_GB} ]]; then
    echo "ERROR: Requested executors (${EXECUTORS}x${EXECUTOR_CORES} cores, ${EXECUTOR_MEMORY_GB}g + overhead) do not fit into workers."
    echo "Hint: reduce --executors/--executor-cores/--executor-memory-gb OR increase SPARK_WORKER_CORES/SPARK_WORKER_MEMORY in docker-compose."
    exit 3
  fi
fi

# ----- Prepare input if needed -----
if [[ "$SKIP_PREPARE" != "1" ]]; then
  echo "[2] Prepare input in HDFS (SCALE=${SCALE})…"
  docker exec -i namenode bash -lc "
    set -euo pipefail
    hdfs dfs -ls -h /HiBench/Wordcount/Input || true
  "
else
  echo "[2] SKIP_PREPARE=${SKIP_PREPARE} — use existing input…"
fi

# ----- Write meta JSON -----
echo "[2.5] Write meta params into container…"
docker exec -i hibench bash -lc "cat > /tmp/wc_meta.json" <<JSON
{
  "benchmark": "wordcount",
  "engine": "${ENGINE}",
  "scale": "${SCALE}",
  "input_gb": ${INPUT_GB},
  "runs": ${RUNS},
  "is_default": ${IS_DEFAULT},
  "executors": ${EXECUTORS},
  "executor_cores": ${EXECUTOR_CORES},
  "executor_memory_gb": ${EXECUTOR_MEMORY_GB},
  "driver_cores": ${DRIVER_CORES},
  "driver_memory_gb": ${DRIVER_MEMORY_GB},
  "parallelism": ${PARALLELISM},
  "shuffle_partitions": ${SHUFFLE_PARTITIONS},
  "rdd_compress": "${RDD_COMPRESS}",
  "io_compression_codec": "${IO_COMPRESSION_CODEC}",
  "spark_conf": "$(echo ${SPARK_CONF} | sed 's/\"/\\\"/g')"
}
JSON

# ----- Run WordCount (multiple) -----
echo "[3] Run WordCount x${RUNS} (SCALE=${SCALE})…"
set +e
docker exec \
  -e RUNS="${RUNS}" \
  -e DATASIZE="${SCALE}" \
  -e SPARK_CONF="${SPARK_CONF}" \
  -e SKIP_PREPARE="${SKIP_PREPARE}" \
  -i hibench bash -lc '
    set -euo pipefail
    export JAVA_HOME=/opt/bitnami/java
    export PATH="$JAVA_HOME/bin:/opt/bitnami/spark/bin:/opt/hadoop/bin:$PATH"
    wordcount_runs_java.sh
  '
rc=$?
set -e

if [[ $rc -ne 0 ]]; then
  echo "!! WordCount run failed (rc=$rc). Diagnostics:"
  docker exec -i hibench bash -lc 'ls -lh /tmp/wc_java_*.log 2>/dev/null || true'
  docker exec -i hibench bash -lc 'tail -n 200 /tmp/wc_java_1.log 2>/dev/null || true'
  docker logs spark-master --tail=200 || true
  exit $rc
fi

# ----- Parse & copy CSVs -----
echo "[4] Parse to CSV (runs + aggregates)…"
docker exec -i hibench bash -lc 'python3 /usr/local/bin/parse_report_to_csv.py'

mkdir -p "${OUT_DIR}"
docker cp hibench:/tmp/wordcount_runs.csv "${OUT_DIR}/wordcount_runs.csv"
docker cp hibench:/tmp/wordcount_agg.csv  "${OUT_DIR}/wordcount_agg.csv"

echo "[OK] CSV saved to: ${OUT_DIR}/wordcount_runs.csv"
echo "[OK] CSV saved to: ${OUT_DIR}/wordcount_agg.csv"
head -n 10 "${OUT_DIR}/wordcount_runs.csv" || true
echo "-----"
cat "${OUT_DIR}/wordcount_agg.csv" || true
