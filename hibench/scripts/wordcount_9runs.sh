#!/usr/bin/env bash
set -euo pipefail
cd /opt/hibench

echo "[*] Prepare WordCount input..."
./bin/workloads/micro/wordcount/prepare/prepare.sh

echo "[*] Run WordCount 9 times..."
for i in $(seq 1 9); do
  echo "[*] Run $i/9..."
  ./bin/workloads/micro/wordcount/spark/run.sh
done

echo "[*] Extract WordCount lines to /tmp/wc_runs.txt"
grep -i wordcount report/hibench.report > /tmp/wc_runs.txt || true

echo "[*] Done. Sample:"
tail -n 5 /tmp/wc_runs.txt || true
