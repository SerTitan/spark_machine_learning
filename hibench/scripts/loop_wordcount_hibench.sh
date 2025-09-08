#!/usr/bin/env bash
set -euo pipefail

# ENV:
#   RUNS, SCALE, SKIP_PREPARE,
#   EXECUTORS, EXECUTOR_CORES, EXECUTOR_MEMORY_GB,
#   DRIVER_CORES, DRIVER_MEMORY_GB,
#   PARALLELISM, SHUFFLE_PARTITIONS, RDD_COMPRESS, IO_COMPRESSION_CODEC,
#   INPUT_GB (опционально, для мета), IS_DEFAULT (0/1)

RUNS="${RUNS:-3}"
SCALE="${SCALE:-tiny}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

EXECUTORS="${EXECUTORS:-2}"
EXECUTOR_CORES="${EXECUTOR_CORES:-1}"
EXECUTOR_MEMORY_GB="${EXECUTOR_MEMORY_GB:-2}"
DRIVER_CORES="${DRIVER_CORES:-1}"
DRIVER_MEMORY_GB="${DRIVER_MEMORY_GB:-1}"
PARALLELISM="${PARALLELISM:-128}"
SHUFFLE_PARTITIONS="${SHUFFLE_PARTITIONS:-128}"
RDD_COMPRESS="${RDD_COMPRESS:-true}"
IO_COMPRESSION_CODEC="${IO_COMPRESSION_CODEC:-lz4}"
INPUT_GB="${INPUT_GB:-8}"
IS_DEFAULT="${IS_DEFAULT:-0}"

export HADOOP_HOME=/opt/hadoop
export HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop
export SPARK_HOME=/opt/bitnami/spark
export PATH=$HADOOP_HOME/bin:$SPARK_HOME/bin:$PATH

# 0) запишем мета для парсера
cat > /tmp/wc_meta.json <<JSON
{
  "benchmark": "wordcount",
  "engine": "spark",
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
  "spark_conf": "--conf spark.executor.instances=${EXECUTORS} --conf spark.executor.cores=${EXECUTOR_CORES} --conf spark.executor.memory=${EXECUTOR_MEMORY_GB}g --conf spark.driver.cores=${DRIVER_CORES} --conf spark.driver.memory=${DRIVER_MEMORY_GB}g --conf spark.default.parallelism=${PARALLELISM} --conf spark.sql.shuffle.partitions=${SHUFFLE_PARTITIONS} --conf spark.rdd.compress=${RDD_COMPRESS} --conf spark.io.compression.codec=${IO_COMPRESSION_CODEC}"
}
JSON

# 1) scale + master для HiBench
sed -i "s/^\(hibench.scale.profile\).*/\1  ${SCALE}/" /opt/hibench/conf/hibench.conf
# Spark оставляем на Standalone мастере:
if grep -q '^hibench.spark.master' /opt/hibench/conf/spark.conf; then
  sed -i 's|^hibench.spark.master.*|hibench.spark.master  spark://spark-master:7077|' /opt/hibench/conf/spark.conf
else
  echo 'hibench.spark.master  spark://spark-master:7077' >> /opt/hibench/conf/spark.conf
fi

# 2) проставим Spark-конфиги в spark.conf (HiBench их подхватит)
sf=/opt/hibench/conf/spark.conf
apply() { key="$1"; val="$2"; grep -q "^${key}[[:space:]]" "$sf" && sed -i "s|^${key}.*|${key}  ${val}|" "$sf" || echo "${key}  ${val}" >> "$sf"; }
apply "spark.executor.instances" "${EXECUTORS}"
apply "spark.executor.cores" "${EXECUTOR_CORES}"
apply "spark.executor.memory" "${EXECUTOR_MEMORY_GB}g"
apply "spark.driver.cores" "${DRIVER_CORES}"
apply "spark.driver.memory" "${DRIVER_MEMORY_GB}g"
apply "spark.default.parallelism" "${PARALLELISM}"
apply "spark.sql.shuffle.partitions" "${SHUFFLE_PARTITIONS}"
apply "spark.rdd.compress" "${RDD_COMPRESS}"
apply "spark.io.compression.codec" "${IO_COMPRESSION_CODEC}"

# 3) prepare через HiBench (MapReduce на YARN)
if [[ "$SKIP_PREPARE" != "1" ]]; then
  echo "[*] HiBench prepare (WordCount, ${SCALE}) on YARN..."
  /opt/hibench/bin/workloads/micro/wordcount/prepare/prepare.sh
fi

# 4) многократные прогоны Spark WordCount
: > /tmp/wc_runs.txt
for i in $(seq 1 "${RUNS}"); do
  echo "[*] Run ${i}/${RUNS}..."
  /opt/hibench/bin/workloads/micro/wordcount/spark/run.sh

  # возьмём последнюю строку WordCount Spark из отчёта
  line=$(grep -E '^WordCount[[:space:]]+Spark[[:space:]]+' /opt/hibench/report/hibench.report | tail -1 || true)
  dur=$(echo "$line" | awk '{print $4}')
  scale=$(echo "$line" | awk '{print $3}')
  if [[ -n "${dur:-}" ]]; then
    printf "WordCount,Spark,%s,%.3f\n" "${scale:-$SCALE}" "${dur}" >> /tmp/wc_runs.txt
  else
    echo "WARN: cannot parse duration from hibench.report" >&2
  fi
done

# 5) CSV (runs + agg)
python3 /usr/local/bin/parse_report_to_csv.py
echo "[*] Done."
