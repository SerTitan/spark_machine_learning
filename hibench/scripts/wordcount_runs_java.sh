#!/usr/bin/env bash
set -euo pipefail

# Входные переменные (передаются извне):
#   RUNS, DATASIZE, SPARK_CONF, SKIP_PREPARE
# Метаданные от раннера: /tmp/wc_meta.json (там input_gb, parallelism и пр.)

META=/tmp/wc_meta.json
RUNS="${RUNS:-3}"
DATASIZE="${DATASIZE:-tiny}"
SPARK_CONF="${SPARK_CONF:-}"
SKIP_PREPARE="${SKIP_PREPARE:-1}"

export JAVA_HOME=/opt/bitnami/java
export HADOOP_HOME=/opt/hadoop
export SPARK_HOME=/opt/bitnami/spark
export PATH="$JAVA_HOME/bin:$HADOOP_HOME/bin:$SPARK_HOME/bin:$PATH"

SPARK_EXAMPLES_JAR="$(ls -1 /opt/bitnami/spark/examples/jars/spark-examples_2.12-*.jar | head -n1)"
INPUT_DIR="hdfs://namenode:8020/HiBench/Wordcount/Input"

# ---- читаем мету ----
INPUT_GB=1
PARALLELISM=128
if [[ -f "$META" ]]; then
  INPUT_GB=$(python3 - <<PY
import json;print(int(json.load(open("$META")).get("input_gb",1)))
PY
)
  PARALLELISM=$(python3 - <<PY
import json;print(int(json.load(open("$META")).get("parallelism",128)))
PY
)
fi

# ---- подготовка данных (настоящий large через RandomTextWriter) ----
if [[ "${SKIP_PREPARE}" != "1" ]]; then
  echo "[*] PREPARE: purge old input & generate ${INPUT_GB} GB via RandomTextWriter → ${INPUT_DIR}"
  hdfs dfs -rm -r -f "${INPUT_DIR}" >/dev/null 2>&1 || true
  hdfs dfs -mkdir -p "$(dirname "${INPUT_DIR}")"

  # Байт в ГБ
  TOTAL_BYTES=$(( INPUT_GB * 1024 * 1024 * 1024 ))

  # Кол-во карт (не слишком мелко/крупно): ~= PARALLELISM/4 в [4..64]
  MAPS=$(( PARALLELISM / 4 ))
  [[ ${MAPS} -lt 4  ]] && MAPS=4
  [[ ${MAPS} -gt 64 ]] && MAPS=64

  # bytes per map (>=128MB)
  BYTES_PER_MAP=$(( TOTAL_BYTES / MAPS ))
  [[ ${BYTES_PER_MAP} -lt 134217728 ]] && BYTES_PER_MAP=134217728

  EX_JAR="$(ls -1 $HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-*.jar | head -n1)"
  if [[ -z "${EX_JAR}" || ! -f "${EX_JAR}" ]]; then
    echo "ERROR: hadoop-mapreduce-examples jar not found under $HADOOP_HOME/share/hadoop/mapreduce" >&2
    exit 5
  fi

  hadoop jar "${EX_JAR}" randomtextwriter \
    -D mapreduce.randomtextwriter.totalbytes="${TOTAL_BYTES}" \
    -D mapreduce.randomtextwriter.bytespermap="${BYTES_PER_MAP}" \
    -D mapreduce.job.maps="${MAPS}" \
    -D mapreduce.job.reduces=0 \
    "${INPUT_DIR}"

  echo "[*] PREPARE: done ($(hdfs dfs -du -h "${INPUT_DIR}" | awk '{s+=$1} END {print (s? s : 0) " bytes total"}'))"
else
  echo "[*] SKIP_PREPARE=1 — пропускаю генерацию, использую существующий ${INPUT_DIR}"
fi

# ---- прогоняем JavaWordCount RUNS раз ----
: > /tmp/wc_runs.txt
for i in $(seq 1 "${RUNS}"); do
  echo "[*] Run ${i}/${RUNS} ..."
  LOG="/tmp/wc_java_${i}.log"
  START=$(date +%s)

  set +e
  spark-submit \
    --master spark://spark-master:7077 \
    ${SPARK_CONF} \
    --class org.apache.spark.examples.JavaWordCount \
    "${SPARK_EXAMPLES_JAR}" \
    "${INPUT_DIR}" \
    > "${LOG}" 2>&1
  rc=$?
  set -e

  END=$(date +%s)
  DUR=$(awk -v s="$START" -v e="$END" 'BEGIN{print (e-s)+0.000}')

  if [[ $rc -ne 0 ]]; then
    echo "!! FAIL run ${i} (rc=${rc}) — смотрим ${LOG}"
    tail -n 200 "${LOG}" || true
    exit $rc
  fi

  printf "WordCount,Spark,%s,%.3f\n" "${DATASIZE}" "${DUR}" | tee -a /tmp/wc_runs.txt
done

echo "[*] Done. Tail wc_runs:"
tail -n 5 /tmp/wc_runs.txt || true
