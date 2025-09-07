#!/usr/bin/env bash
set -euo pipefail

echo "[0] Bringing full stack up…"
docker compose up -d

echo "[1] Waiting for HDFS (namenode RPC 8020) to respond…"
for i in {1..60}; do
  if docker exec namenode /opt/hadoop/bin/hdfs dfs -ls / >/dev/null 2>&1; then
    echo "    HDFS on namenode is OK"; break
  fi
  sleep 2
  if [ $i -eq 60 ]; then
    echo "!! HDFS failed to come up on namenode"
    docker compose logs --tail=200 namenode datanode
    exit 1
  fi
done

echo "[2] Waiting for Spark Master UI (http://localhost:8080)…"
for _ in {1..60}; do
  if curl -fsS http://localhost:8080 >/dev/null 2>&1 || wget -qO- http://localhost:8080 >/dev/null 2>&1; then
    echo "    Spark Master UI OK"; break
  fi
  sleep 2
done

echo "[3] Waiting for Spark History UI (http://localhost:18080)…"
for _ in {1..60}; do
  if curl -fsS http://localhost:18080 >/dev/null 2>&1 || wget -qO- http://localhost:18080 >/dev/null 2>&1; then
    echo "    Spark History UI OK"; break
  fi
  sleep 2
done

echo "[4] Sanity checks (spark/hdfs from containers)…"
docker exec -it spark-master bash -lc '/opt/bitnami/spark/bin/spark-submit --version | head -n 1 || true'
docker exec -it hibench       bash -lc '/opt/hadoop/bin/hdfs dfs -fs hdfs://namenode:8020 -ls / | head || true'

echo "[OK] Stack is up."
