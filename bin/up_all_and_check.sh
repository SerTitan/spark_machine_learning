#!/usr/bin/env bash
set -euo pipefail

echo "[0] Bringing full stack up…"
docker compose up -d

# ---------- HDFS readyness: проверяем прямо на NN ----------
echo "[1] Waiting for HDFS (namenode RPC 8020) to respond…"
for i in {1..60}; do
  if docker exec namenode hdfs dfs -ls / >/dev/null 2>&1; then
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
  if [ $i -eq 60 ]; then
    echo "!! Spark Master UI not responding"
    docker compose logs --tail=200 spark-master
    exit 1
  fi
done

# ---------- History UI ----------
echo "[3] Waiting for Spark History UI (http://localhost:18080)…"
for i in {1..60}; do
  if curl -fsS http://localhost:18080 >/dev/null 2>&1 || wget -qO- http://localhost:18080 >/dev/null 2>&1; then
    echo "    Spark History UI OK"; break
  fi
  sleep 2
  if [ $i -eq 60 ]; then
    echo "!! Spark History UI not responding"
    docker compose logs --tail=200 spark-history
    exit 1
  fi
done

# ---------- Авто-фиксы HiBench ----------
echo "[3.5] Auto-fix HiBench inside container…"
if [[ -x "./bin/hibench_auto_fix.sh" ]]; then
  ./bin/hibench_auto_fix.sh
else
  echo "!! bin/hibench_auto_fix.sh not found or not executable"; exit 1
fi

# ---------- HDFS smoke из hibench (с явной средой Java+PATH) ----------
echo "[4] HDFS client smoke test from hibench…"
docker exec hibench bash -lc '
  set -e
  export JAVA_HOME=/opt/java8
  export PATH=/opt/java8/bin:/opt/hadoop/bin:/opt/hadoop/sbin:$PATH
  echo "hello hdfs" > /tmp/hello.txt
  hdfs dfs -D fs.defaultFS=hdfs://namenode:8020 -mkdir -p /bench/input
  hdfs dfs -D fs.defaultFS=hdfs://namenode:8020 -put -f /tmp/hello.txt /bench/input/hello.txt
  test "$(hdfs dfs -D fs.defaultFS=hdfs://namenode:8020 -cat /bench/input/hello.txt)" = "hello hdfs"
'
echo "    Hibench HDFS client OK"

# ---------- Spark Pi ----------
echo "[5] Running Spark Pi (to generate an event log for History)…"
docker exec hibench bash -lc '
  export JAVA_HOME=/opt/bitnami/java
  export PATH=/opt/bitnami/java/bin:/opt/bitnami/spark/bin:$PATH
  export PYSPARK_PYTHON=/opt/bitnami/python/bin/python3
  export PYSPARK_DRIVER_PYTHON=/opt/bitnami/python/bin/python3
  /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --conf spark.eventLog.enabled=true \
    --conf spark.eventLog.dir=file:/opt/spark/history \
    --conf spark.executor.instances=2 \
    --conf spark.executor.cores=1 \
    --conf spark.pyspark.python=/opt/bitnami/python/bin/python3 \
    --conf spark.pyspark.driver.python=/opt/bitnami/python/bin/python3 \
    /opt/spark/examples/src/main/python/pi.py 50
' | tee /tmp/spark_pi.out

grep -E "Pi is roughly|Estimated value of Pi" /tmp/spark_pi.out >/dev/null || { echo "!! Spark Pi output not found"; exit 1; }

# ---------- History API ----------
echo "[6] Checking History Server API…"
if curl -fsS http://localhost:18080/api/v1/applications | grep -q '"id"'; then
  echo "    History Server sees at least one app ✔"
else
  echo "    WARN: History API empty (try refreshing UI in a few seconds)"
fi

echo "✅ All good: HDFS + Spark + History are up."
echo "   UIs: NN http://localhost:9870  |  Master http://localhost:8080  |  History http://localhost:18080"
