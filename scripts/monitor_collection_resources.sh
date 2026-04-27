#!/usr/bin/env bash
set -euo pipefail
#
# monitor_collection_resources.sh — фоновый мониторинг ресурсов VM во время сбора.
#
# Раз в INTERVAL секунд (по умолчанию 300) пишет в logs/ snapshot:
#   df -h, free -m, docker stats --no-stream, du -sh data out logs.
# Используется параллельно с длительным сбором, чтобы по логам понять,
# не упёрлись ли в диск или RAM, и где именно растёт потребление.
#

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INTERVAL="${INTERVAL:-300}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/collection_resources_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$LOG_DIR"

snapshot() {
  local ts
  ts="$(date --iso-8601=seconds)"

  {
    echo
    echo "===== ${ts} ====="
    echo "--- host df ---"
    df -h "$PROJECT_DIR" /var/lib/docker 2>/dev/null || df -h "$PROJECT_DIR"

    echo "--- host memory ---"
    free -h || true

    echo "--- docker system df ---"
    docker system df 2>/dev/null || echo "docker system df unavailable"

    echo "--- project data/out/logs size ---"
    du -sh "$PROJECT_DIR/data" "$PROJECT_DIR/out" "$PROJECT_DIR/logs" 2>/dev/null || true

    echo "--- running containers ---"
    docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || true

    echo "--- hdfs usage ---"
    docker exec hibench bash -lc '/opt/hadoop/bin/hdfs dfs -du -h / 2>/dev/null | sort -h || true' 2>/dev/null || true
  } | tee -a "$LOG_FILE"
}

echo "Logging resource snapshots to: ${LOG_FILE}"
echo "Interval: ${INTERVAL}s"

while true; do
  snapshot
  sleep "$INTERVAL"
done
