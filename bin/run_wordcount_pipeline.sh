#!/usr/bin/env bash
set -euo pipefail

# ===== ПАРАМЕТРЫ =====
RUNS="${RUNS:-9}"
NUM_LINES="${NUM_LINES:-500000}"
SCALE="${SCALE:-tiny}"
OUT_CSV="${OUT_CSV:-./out/wordcount_runs.csv}"
SPARK_CONF="${SPARK_CONF:-}"  # напр.: '--conf spark.executor.cores=2 --conf spark.executor.memory=2g'

echo "[0] Build & up (hibench)…"
docker compose build hibench >/dev/null
docker compose up -d

echo "[1] Health checks…"
./bin/up_all_and_check.sh

echo "[2] Run WordCount x$RUNS (NUM_LINES=$NUM_LINES, SCALE=$SCALE)…"
docker exec -it hibench bash -lc \
  "RUNS='$RUNS' NUM_LINES='$NUM_LINES' DATASIZE='$SCALE' SPARK_CONF=\"$SPARK_CONF\" wordcount_runs_java.sh"

echo "[3] Parse to CSV…"
docker exec -it hibench bash -lc 'python3 /usr/local/bin/parse_report_to_csv.py'

mkdir -p "$(dirname "$OUT_CSV")"
docker cp hibench:/tmp/wordcount_runs.csv "$OUT_CSV"

echo "[OK] CSV saved to: $OUT_CSV"
tail -n +1 "$OUT_CSV"
