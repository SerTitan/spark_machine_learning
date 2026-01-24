#!/usr/bin/env bash
set -euo pipefail
echo "[*] Spark Master UI:"; curl -sSf http://localhost:8080 >/dev/null && echo "  OK"
echo "[*] NameNode UI:";     curl -sSf http://localhost:9870 >/dev/null && echo "  OK"
echo "[*] Spark History:";   curl -sSf http://localhost:18080 >/dev/null && echo "  OK"
docker exec -it spark-master bash -lc 'spark-submit --version && echo "spark ok"'
docker exec -it namenode bash -lc  'hdfs dfs -ls / || true'
