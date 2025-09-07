#!/usr/bin/env bash
set -euo pipefail

echo "[0] Bringing full stack up…"
docker compose up -d

# ---------- HDFS readyness ----------
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

# ---------- Spark Master UI ----------
echo "[2] Waiting for Spark Master UI (http://localhost:8080)…"
for i in {1..60}; do
  if curl -fsS http://localhost:8080 >/dev/null 2>&1 || wget -qO- http://localhost:8080 >/dev/null 2>&1; then
    echo "    Spark Master UI OK"; break
  fi
  sleep 2
done

# ---------- Spark History UI ----------
echo "[3] Waiting for Spark History UI (http://localhost:18080)…"
for i in {1..60}; do
  if curl -fsS http://localhost:18080 >/dev/null 2>&1 || wget -qO- http://localhost:18080 >/dev/null 2>&1; then
    echo "    Spark History UI OK"; break
  fi
  sleep 2
done

# ---------- Sanity checks ----------
echo "[4] Sanity checks (spark/hdfs from containers)…"

# Spark in bitnami image lives in /opt/bitnami/spark
if ! docker exec spark-master bash -lc '/opt/bitnami/spark/bin/spark-submit --version | head -n 1'; then
  echo "!! spark-submit not found at /opt/bitnami/spark/bin/spark-submit inside spark-master"
  echo "   PATH inside spark-master:"
  docker exec spark-master bash -lc 'echo $PATH'
fi

# Hadoop client in hibench image is /opt/hadoop/bin/hdfs
if ! docker exec hibench bash -lc '/opt/hadoop/bin/hdfs dfs -fs hdfs://namenode:8020 -ls / | head -n 5'; then
  echo "!! hdfs not found at /opt/hadoop/bin/hdfs inside hibench"
  echo "   PATH inside hibench:"
  docker exec hibench bash -lc 'echo $PATH; ls -l /opt/hadoop/bin | sed -n "1,20p"'
fi

echo "[OK] Stack is up."
