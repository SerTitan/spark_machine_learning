#!/usr/bin/env bash
set -euo pipefail

# ===== ПАРАМЕТРЫ =====
RUNS="${RUNS:-9}"
NUM_LINES="${NUM_LINES:-500000}"
SCALE="${SCALE:-tiny}"
OUT_CSV="${OUT_CSV:-./out/wordcount_runs.csv}"
# Безопасные дефолты (можно переопределить снаружи)
SPARK_CONF="${SPARK_CONF:---conf spark.executor.cores=1 --conf spark.executor.memory=512m}"
SKIP_PREPARE="${SKIP_PREPARE:-1}"    # 1=не готовим вход; 0=перегенерим вход

echo "[0] Build & up (hibench)…"
docker compose build hibench >/dev/null
docker compose up -d

echo "[1] Health checks…"
./bin/up_all_and_check.sh

# === вместо сложного ожидания воркеров — короткая пауза (как в ручном запуске) ===
echo "[1.5] Short settle sleep before submitting jobs…"
sleep 5

# === (опц.) подготовка входа в HDFS через namenode ===
if [[ "$SKIP_PREPARE" != "1" ]]; then
  echo "[2] Prepare input in HDFS via namenode (NUM_LINES=${NUM_LINES})…"
  docker exec -i namenode bash -lc "
    set -euo pipefail
    yes 'lorem ipsum dolor sit amet consectetur adipiscing elit' | head -n ${NUM_LINES} > /tmp/wc_input.txt
    hdfs dfs -rm -r -f -skipTrash /HiBench/Wordcount/Input || true
    hdfs dfs -mkdir -p /HiBench/Wordcount/Input
    hdfs dfs -put -f /tmp/wc_input.txt /HiBench/Wordcount/Input/part-00000
    hdfs dfs -ls -h /HiBench/Wordcount/Input | sed -n '1,5p'
  "
else
  echo "[2] SKIP_PREPARE=1 — пропускаем подготовку (используем существующий вход)…"
fi

# === запуск бенча (ровно как твоя рабочая команда) ===
echo "[3] Run WordCount x${RUNS} (SCALE=${SCALE}, SKIP_PREPARE=${SKIP_PREPARE})…"
set +e
docker exec \
  -e RUNS="${RUNS}" \
  -e NUM_LINES="${NUM_LINES}" \
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
  echo "!! WordCount run failed (rc=$rc). Quick diagnostics:"
  docker exec -i hibench bash -lc 'ls -lh /tmp/wc_java_*.log 2>/dev/null || true'
  docker exec -i hibench bash -lc 'tail -n 120 /tmp/wc_java_1.log 2>/dev/null || true'
  docker logs spark-master --tail=150 || true
  exit $rc
fi

# === парсинг и выгрузка CSV ===
echo "[4] Parse to CSV…"
docker exec -i hibench bash -lc 'python3 /usr/local/bin/parse_report_to_csv.py'

mkdir -p "$(dirname "$OUT_CSV")"
docker cp hibench:/tmp/wordcount_runs.csv "$OUT_CSV"

echo "[OK] CSV saved to: $OUT_CSV"
tail -n +1 "$OUT_CSV"
