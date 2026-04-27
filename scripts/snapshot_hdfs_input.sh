#!/usr/bin/env bash
set -euo pipefail
#
# snapshot_hdfs_input.sh — архивация HDFS-входа в tar.gz на хост-диск.
#
# После prepare HiBench кладёт сгенерированные данные в HDFS. Чтобы при
# следующих запусках не делать prepare заново (а это для PageRank — десятки
# минут), забираем содержимое HDFS-пути и складываем рядом с проектом в
# data/hdfs_inputs/<job_type>/<profile>/<dataset_id>.tar.gz.
# Пара со scripts/restore_hdfs_input.sh.
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-${PROJECT_DIR}/data/hdfs_inputs}"

JOB_TYPE="${JOB_TYPE:?JOB_TYPE is required}"
PROFILE="${PROFILE:?PROFILE is required}"
INPUT_DATASET_ID="${INPUT_DATASET_ID:?INPUT_DATASET_ID is required}"
HDFS_PATH="${HDFS_PATH:?HDFS_PATH is required}"

SAFE_ID="$(echo "$INPUT_DATASET_ID" | tr -c 'A-Za-z0-9_.-' '_')"
OUT_DIR="${SNAPSHOT_ROOT}/${JOB_TYPE}/${PROFILE}"
ARCHIVE_NAME="${SAFE_ID}.tar.gz"
ARCHIVE_PATH="${OUT_DIR}/${ARCHIVE_NAME}"
META_PATH="${OUT_DIR}/${SAFE_ID}.metadata.json"
CONTAINER_TMP="/tmp/${ARCHIVE_NAME}"
CONTAINER_EXTRACT_DIR="/tmp/hdfs_snapshot_${SAFE_ID}"

mkdir -p "$OUT_DIR"

echo "=== Snapshot HDFS input ==="
echo "JOB_TYPE=${JOB_TYPE}"
echo "PROFILE=${PROFILE}"
echo "INPUT_DATASET_ID=${INPUT_DATASET_ID}"
echo "HDFS_PATH=${HDFS_PATH}"
echo "ARCHIVE_PATH=${ARCHIVE_PATH}"

docker exec hibench bash -lc "/opt/hadoop/bin/hdfs dfs -test -e '${HDFS_PATH}'"

input_size_bytes="$(docker exec hibench bash -lc "/opt/hadoop/bin/hdfs dfs -du -s '${HDFS_PATH}' | awk '{print \$1}'" | tr -d '\r')"
generated_at="$(date --iso-8601=seconds)"

docker exec hibench bash -lc "rm -rf '${CONTAINER_EXTRACT_DIR}' '${CONTAINER_TMP}' && mkdir -p '${CONTAINER_EXTRACT_DIR}'"
docker exec hibench bash -lc "/opt/hadoop/bin/hdfs dfs -get '${HDFS_PATH}' '${CONTAINER_EXTRACT_DIR}/input'"
docker exec hibench bash -lc "cd '${CONTAINER_EXTRACT_DIR}' && tar -czf '${CONTAINER_TMP}' input"
docker cp "hibench:${CONTAINER_TMP}" "$ARCHIVE_PATH"
docker exec hibench bash -lc "rm -rf '${CONTAINER_EXTRACT_DIR}' '${CONTAINER_TMP}'"

sha256="$(sha256sum "$ARCHIVE_PATH" | awk '{print $1}')"
archive_size_bytes="$(stat -c '%s' "$ARCHIVE_PATH")"

cat > "$META_PATH" <<EOF
{
  "job_type": "${JOB_TYPE}",
  "profile": "${PROFILE}",
  "input_dataset_id": "${INPUT_DATASET_ID}",
  "hdfs_path": "${HDFS_PATH}",
  "snapshot_archive": "${ARCHIVE_PATH}",
  "input_size_bytes": ${input_size_bytes:-0},
  "archive_size_bytes": ${archive_size_bytes},
  "sha256": "${sha256}",
  "generated_at": "${generated_at}"
}
EOF

echo ">>> Snapshot archive: ${ARCHIVE_PATH}"
echo ">>> Metadata: ${META_PATH}"
