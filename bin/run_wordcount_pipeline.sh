#!/usr/bin/env bash
set -euo pipefail

# ===== Defaults (можно переопределять окружением или флагами) =====
RUNS="${RUNS:-9}"
SCALE="${SCALE:-tiny}"               # HiBench scale label
OUT_DIR="${OUT_DIR:-./out}"          # куда скопировать CSV на хосте
SKIP_PREPARE="${SKIP_PREPARE:-1}"    # 1=не готовим вход; 0=перегенерим вход

# Ключевые Spark-параметры (набор можно расширять)
EXECUTORS="${EXECUTORS:-2}"
EXECUTOR_CORES="${EXECUTOR_CORES:-2}"
EXECUTOR_MEMORY_GB="${EXECUTOR_MEMORY_GB:-4}"     # целое, в ГБ
DRIVER_CORES="${DRIVER_CORES:-2}"
DRIVER_MEMORY_GB="${DRIVER_MEMORY_GB:-4}"         # целое, в ГБ
PARALLELISM="${PARALLELISM:-256}"
SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-256}"
RDD_COMPRESS="${RDD_COMPRESS:-true}"               # true/false
IO_COMPRESSION_CODEC="${IO_COMPRESSION_CODEC:-lz4}" # lz4/snappy/zstd
INPUT_GB="${INPUT_GB:-1}"                          # для отчётов и фичей
ENGINE="${ENGINE:-spark}"                           # метка движка

# ----- Парсим флаги (опц.) -----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs) RUNS="$2"; shift 2;;
    --scale) SCALE="$2"; shift 2;;
    --outdir|--out-dir) OUT_DIR="$2"; shift 2;;
    --skip-prepare) SKIP_PREPARE="$2"; shift 2;;

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

# Сборка SPARK_CONF из параметров
SPARK_CONF="--conf spark.executor.instances=${EXECUTORS} \
--conf spark.executor.cores=${EXECUTOR_CORES} \
--conf spark.executor.memory=${EXECUTOR_MEMORY_GB}g \
--conf spark.driver.cores=${DRIVER_CORES} \
--conf spark.driver.memory=${DRIVER_MEMORY_GB}g \
--conf spark.default.parallelism=${PARALLELISM} \
--conf spark.sql.shuffle.partitions=${SHUFFLE_PARTITIONS} \
--conf spark.rdd.compress=${RDD_COMPRESS} \
--conf spark.io.compression.codec=${IO_COMPRESSION_CODEC}"

echo "[0] Build & up (hibench)…"
docker compose build hibench >/dev/null
docker compose up -d

echo "[1] Health checks…"
./bin/up_all_and_check.sh

echo "[1.5] Short settle sleep before submitting jobs…"
sleep 5

# ----- Подготовка входа (если требуется) -----
if [[ "$SKIP_PREPARE" != "1" ]]; then
  echo "[2] Prepare input in HDFS via namenode (SCALE=${SCALE})…"
  docker exec -i namenode bash -lc "
    set -euo pipefail
    # Здесь можно сделать генерацию входа под SCALE/INPUT_GB (упрощенно оставим как было)
    hdfs dfs -ls -h /HiBench/Wordcount/Input || true
  "
else
  echo "[2] SKIP_PREPARE=${SKIP_PREPARE} — используем существующий вход…"
fi

# ----- Создаём meta JSON с параметрами запуска -----
echo "[2.5] Write meta (Spark params) into container…"
docker exec -i hibench bash -lc "cat > /tmp/wc_meta.json" <<JSON
{
  "benchmark": "wordcount",
  "engine": "${ENGINE}",
  "scale": "${SCALE}",
  "input_gb": ${INPUT_GB},
  "runs": ${RUNS},
  "executors": ${EXECUTORS},
  "executor_cores": ${EXECUTOR_CORES},
  "executor_memory_gb": ${EXECUTOR_MEMORY_GB},
  "driver_cores": ${DRIVER_CORES},
  "driver_memory_gb": ${DRIVER_MEMORY_GB},
  "parallelism": ${PARALLELISM},
  "shuffle_partitions": ${SHUFFLE_PARTITIONS},
  "rdd_compress": "${RDD_COMPRESS}",
  "io_compression_codec": "${IO_COMPRESSION_CODEC}",
  "spark_conf": "$(echo ${SPARK_CONF} | sed 's/"/\\"/g')"
}
JSON

# ----- Запуск WordCount (множественно) -----
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
    # ожидается, что wordcount_runs_java.sh записывает /tmp/wc_runs.txt и/или обновляет hibench.report
    wordcount_runs_java.sh
  '
rc=$?
set -e

if [[ $rc -ne 0 ]]; then
  echo "!! WordCount run failed (rc=$rc). Diagnostics:"
  docker exec -i hibench bash -lc 'ls -lh /tmp/wc_java_*.log 2>/dev/null || true'
  docker exec -i hibench bash -lc 'tail -n 120 /tmp/wc_java_1.log 2>/dev/null || true'
  docker logs spark-master --tail=150 || true
  exit $rc
fi

# ----- Парсинг и выгрузка CSV -----
echo "[4] Parse to CSV (runs + aggregates)…"
docker exec -i hibench bash -lc 'python3 /usr/local/bin/parse_report_to_csv.py'

mkdir -p "${OUT_DIR}"
docker cp hibench:/tmp/wordcount_runs.csv "${OUT_DIR}/wordcount_runs.csv"
docker cp hibench:/tmp/wordcount_agg.csv  "${OUT_DIR}/wordcount_agg.csv"

echo "[OK] CSV saved to: ${OUT_DIR}/wordcount_runs.csv"
echo "[OK] CSV saved to: ${OUT_DIR}/wordcount_agg.csv"
tail -n +1 "${OUT_DIR}/wordcount_runs.csv" | sed -n '1,10p'
echo "-----"
tail -n +1 "${OUT_DIR}/wordcount_agg.csv"
