#!/usr/bin/env bash
set -euo pipefail

# ===== Базовое окружение =====
export HIBENCH_HOME=${HIBENCH_HOME:-/opt/hibench}
export WORKLOAD_RESULT_FOLDER=${WORKLOAD_RESULT_FOLDER:-$HIBENCH_HOME/report}
export SPARK_HOME=${SPARK_HOME:-/opt/spark}
export HADOOP_HOME=${HADOOP_HOME:-/opt/hadoop}

# >>> Критично: используем встроенную Java из bitnami (Java 17), а не Java 8 <<<
if [ -d /opt/bitnami/java ]; then
  export JAVA_HOME=/opt/bitnami/java
  export PATH="$JAVA_HOME/bin:$PATH"
fi

# Добавим spark/hadoop в PATH
export PATH="/opt/bitnami/spark/bin:$SPARK_HOME/bin:$HADOOP_HOME/bin:$PATH"

# Явно указываем NameNode для всех HDFS-команд
HDFS_URI="${HDFS_URI:-hdfs://namenode:8020}"

# Константы
SCALE="${DATASIZE:-tiny}"
HDFS_IN_DIR="/HiBench/Wordcount/Input"
HDFS_IN_PATH="$HDFS_IN_DIR/part-00000"
LOCAL_TMP="/tmp/wc_input.txt"
REPORT="$WORKLOAD_RESULT_FOLDER/hibench.report"
RUNS_FILE="/tmp/wc_runs.txt"
RUNS="${RUNS:-9}"

# Определяем spark-submit
if [ -x /opt/bitnami/spark/bin/spark-submit ]; then
  SPARK_SUBMIT="/opt/bitnami/spark/bin/spark-submit"
elif [ -x /opt/spark/bin/spark-submit ]; then
  SPARK_SUBMIT="/opt/spark/bin/spark-submit"
else
  SPARK_SUBMIT="$(command -v spark-submit || true)"
fi
if [ -z "${SPARK_SUBMIT:-}" ]; then
  echo "[ERR] spark-submit not found" >&2
  exit 1
fi

echo "[*] Using JAVA_HOME=$JAVA_HOME"
echo "[*] Using spark-submit at: $SPARK_SUBMIT"

# ===== Проверки =====
echo "[*] Checking HDFS connectivity to $HDFS_URI ..."
if ! hdfs dfs -fs "$HDFS_URI" -ls / >/dev/null 2>&1; then
  echo "[ERR] Cannot list HDFS root via -fs $HDFS_URI" >&2
  exit 1
fi
echo "    HDFS OK."

# ===== Подготовка директорий =====
mkdir -p "$WORKLOAD_RESULT_FOLDER"
rm -f "$RUNS_FILE"
if [ ! -f "$REPORT" ]; then
  echo -e "Benchmark\tEngine\tScale\tDuration(s)" > "$REPORT"
fi

# ===== Генерация входных данных и загрузка в HDFS =====
echo "[*] Prepare WordCount input (generate -> put to HDFS $HDFS_IN_PATH) ..."
NUM_LINES=${NUM_LINES:-500000}
yes "lorem ipsum dolor sit amet consectetur adipiscing elit" | head -n "$NUM_LINES" > "$LOCAL_TMP"

hdfs dfs -fs "$HDFS_URI" -rm -r -f -skipTrash "$HDFS_IN_DIR" >/dev/null 2>&1 || true
hdfs dfs -fs "$HDFS_URI" -mkdir -p "$HDFS_IN_DIR"
hdfs dfs -fs "$HDFS_URI" -put -f "$LOCAL_TMP" "$HDFS_IN_PATH"

if ! hdfs dfs -fs "$HDFS_URI" -test -f "$HDFS_IN_PATH"; then
  echo "[ERR] Input not found in HDFS at $HDFS_IN_PATH" >&2
  exit 1
fi
echo "    HDFS input ready: $HDFS_IN_PATH"

# ===== Подготовка PySpark WordCount (надёжный вариант) =====
PYJOB="/tmp/wc_job.py"
cat > "$PYJOB" <<'PY'
import sys
from pyspark.sql import SparkSession

if len(sys.argv) < 2:
    print("Usage: wc_job.py <hdfs_input_dir>")
    sys.exit(2)

inp = sys.argv[1]
spark = SparkSession.builder.appName("PyWordCount").getOrCreate()
sc = spark.sparkContext

rdd = sc.textFile(inp)
# Классический WordCount; count() материализует вычисление
_ = (rdd
     .flatMap(lambda line: line.strip().split())
     .map(lambda w: (w, 1))
     .reduceByKey(lambda a, b: a + b)
     .count())
spark.stop()
PY

# ===== Запуски WordCount =====
echo "[*] Run WordCount $RUNS times via spark-submit (PySpark)…"
for i in $(seq 1 "$RUNS"); do
  echo "[*] Run $i/$RUNS ..."
  LOG="/tmp/wc_run_${i}.log"
  start_ns=$(date +%s%N)

  set +e
  "$SPARK_SUBMIT" \
    --master spark://spark-master:7077 \
    "$PYJOB" "$HDFS_URI$HDFS_IN_DIR" \
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
    echo "[ERR] spark-submit failed on run $i (rc=$rc), last 60 lines:"
    tail -n 60 "$LOG" || true
    exit $rc
  fi

  echo "WordCount,Spark,${SCALE},${dur_sec}" >> "$RUNS_FILE"
  printf "WordCount\tSpark\t%s\t%s\n" "$SCALE" "$dur_sec" >> "$REPORT"
  echo "    OK run $i: ${dur_sec}s"
done

echo "[*] Done. Tail of wc_runs.txt:"
tail -n 5 "$RUNS_FILE" || true

echo "[*] Report at $REPORT ; CSV via parse_report_to_csv.py"
