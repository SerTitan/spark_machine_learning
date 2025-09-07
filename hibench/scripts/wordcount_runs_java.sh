#!/usr/bin/env bash
set -euo pipefail

# ========= ПАРАМЕТРЫ =========
RUNS="${RUNS:-9}"
NUM_LINES="${NUM_LINES:-500000}"
SCALE="${DATASIZE:-tiny}"
HDFS_URI="${HDFS_URI:-hdfs://namenode:8020}"
HDFS_IN_DIR="${HDFS_IN_DIR:-/HiBench/Wordcount/Input}"
HDFS_IN_PATH="$HDFS_IN_DIR/part-00000"
SPARK_CONF="${SPARK_CONF:-}"

# ========= ОКРУЖЕНИЕ =========
export HIBENCH_HOME=${HIBENCH_HOME:-/opt/hibench}
export WORKLOAD_RESULT_FOLDER=${WORKLOAD_RESULT_FOLDER:-$HIBENCH_HOME/report}
export HADOOP_HOME=${HADOOP_HOME:-/opt/hadoop}
export SPARK_HOME=${SPARK_HOME:-/opt/bitnami/spark}
if [ -d /opt/bitnami/java ]; then
  export JAVA_HOME=/opt/bitnami/java
  export PATH="$JAVA_HOME/bin:$PATH"
fi
export PATH="/opt/bitnami/spark/bin:$SPARK_HOME/bin:$HADOOP_HOME/bin:$PATH"

REPORT="$WORKLOAD_RESULT_FOLDER/hibench.report"
RUNS_FILE="/tmp/wc_runs.txt"
LOCAL_TMP="/tmp/wc_input.txt"

# ========= ПРОВЕРКИ =========
command -v /opt/bitnami/spark/bin/spark-submit >/dev/null || { echo "[ERR] spark-submit not found"; exit 1; }
command -v hdfs >/dev/null || { echo "[ERR] hdfs not found"; exit 1; }
if ! hdfs dfs -fs "$HDFS_URI" -ls / >/dev/null 2>&1; then
  echo "[ERR] HDFS is not reachable at $HDFS_URI"; exit 1
fi

JAR="$(ls -1 /opt/bitnami/spark/examples/jars/spark-examples_2.12-3.*.jar 2>/dev/null | head -n1 || true)"
if [ -z "$JAR" ]; then
  echo "[ERR] spark-examples jar not found in /opt/bitnami/spark/examples/jars/"; exit 1
fi

mkdir -p "$WORKLOAD_RESULT_FOLDER"
echo -e "Benchmark\tEngine\tScale\tDuration(s)" > "$REPORT"
: > "$RUNS_FILE"

# ========= ПОДГОТОВКА ВХОДА =========
echo "[*] Preparing input: $NUM_LINES lines -> $HDFS_URI$HDFS_IN_PATH"
yes "lorem ipsum dolor sit amet consectetur adipiscing elit" | head -n "$NUM_LINES" > "$LOCAL_TMP"
hdfs dfs -fs "$HDFS_URI" -rm -r -f -skipTrash "$HDFS_IN_DIR" >/dev/null 2>&1 || true
hdfs dfs -fs "$HDFS_URI" -mkdir -p "$HDFS_IN_DIR"
hdfs dfs -fs "$HDFS_URI" -put -f "$LOCAL_TMP" "$HDFS_IN_PATH"
hdfs dfs -fs "$HDFS_URI" -ls -h "$HDFS_IN_DIR" | sed -n '1,5p'

# ========= ЗАПУСКИ =========
echo "[*] Running JavaWordCount x $RUNS"
for i in $(seq 1 "$RUNS"); do
  echo "[*] Run $i/$RUNS ..."
  LOG="/tmp/wc_java_$i.log"
  start_ns=$(date +%s%N)
  set +e
  /opt/bitnami/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    $SPARK_CONF \
    --class org.apache.spark.examples.JavaWordCount \
    "$JAR" "$HDFS_URI$HDFS_IN_DIR" \
    >"$LOG" 2>&1
  rc=$?
  set -e
  end_ns=$(date +%s%N)
  dur_ms=$(( (end_ns - start_ns) / 1000000 ))
  dur_sec=$(python3 - <<PY
d=$dur_ms/1000.0
print(f"{d:.3f}")
PY
)
  if [ $rc -ne 0 ]; then
    echo "[ERR] Run $i failed (rc=$rc). Tail log:"
    tail -n 80 "$LOG" || true
    exit $rc
  fi
  echo "WordCount,Spark,${SCALE},${dur_sec}" >> "$RUNS_FILE"
  printf "WordCount\tSpark\t%s\t%s\n" "$SCALE" "$dur_sec" >> "$REPORT"
  echo "    OK $i: ${dur_sec}s"
done

echo "[*] Done. Tail wc_runs:"
tail -n 5 "$RUNS_FILE" || true
echo "[*] Report: $REPORT"
