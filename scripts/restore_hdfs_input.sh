#!/usr/bin/env bash
set -euo pipefail
#
# restore_hdfs_input.sh — восстановление HDFS-входа из tar.gz.
#
# Парный к snapshot_hdfs_input.sh. Распаковывает архив и заливает содержимое
# в указанный HDFS-путь, чтобы коллектор смог пропустить prepare.
#

ARCHIVE_PATH="${ARCHIVE_PATH:?ARCHIVE_PATH is required}"
TARGET_HDFS_PATH="${TARGET_HDFS_PATH:?TARGET_HDFS_PATH is required}"
OVERWRITE="${OVERWRITE:-1}"

if [[ ! -f "$ARCHIVE_PATH" ]]; then
  echo "ERROR: archive not found: $ARCHIVE_PATH" >&2
  exit 1
fi

ARCHIVE_BASENAME="$(basename "$ARCHIVE_PATH")"
CONTAINER_TMP="/tmp/${ARCHIVE_BASENAME}"
CONTAINER_EXTRACT_DIR="/tmp/hdfs_restore_${ARCHIVE_BASENAME%.tar.gz}"

echo "=== Restore HDFS input ==="
echo "ARCHIVE_PATH=${ARCHIVE_PATH}"
echo "TARGET_HDFS_PATH=${TARGET_HDFS_PATH}"

docker cp "$ARCHIVE_PATH" "hibench:${CONTAINER_TMP}"
docker exec hibench bash -lc "rm -rf '${CONTAINER_EXTRACT_DIR}' && mkdir -p '${CONTAINER_EXTRACT_DIR}' && tar -xzf '${CONTAINER_TMP}' -C '${CONTAINER_EXTRACT_DIR}'"

if [[ "$OVERWRITE" == "1" ]]; then
  docker exec hibench bash -lc "/opt/hadoop/bin/hdfs dfs -rm -r -f '${TARGET_HDFS_PATH}' >/dev/null 2>&1 || true"
fi

docker exec hibench bash -lc "/opt/hadoop/bin/hdfs dfs -mkdir -p '$(dirname "$TARGET_HDFS_PATH")'"
docker exec hibench bash -lc "/opt/hadoop/bin/hdfs dfs -put -f '${CONTAINER_EXTRACT_DIR}/input' '${TARGET_HDFS_PATH}'"
docker exec hibench bash -lc "rm -rf '${CONTAINER_EXTRACT_DIR}' '${CONTAINER_TMP}'"

docker exec hibench bash -lc "/opt/hadoop/bin/hdfs dfs -du -s '${TARGET_HDFS_PATH}'"
echo ">>> Restored to HDFS: ${TARGET_HDFS_PATH}"
