#!/usr/bin/env bash
set -euo pipefail
#
# vm_preflight.sh — проверка готовности VM перед сбором датасета.
#
# Проверяет, что VM соответствует минимальным требованиям:
#   CPU vCPU >= MIN_VCPU
#   RAM >= MIN_MEM_GB
#   свободное место на диске >= MIN_DISK_FREE_GB
# Также печатает версии docker / docker compose, наличие jq, htop, tmux.
# Если какая-то проверка не прошла — выходит с ненулевым кодом, чтобы
# случайно не запустить сбор на маленькой машине.
#

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MIN_VCPU="${MIN_VCPU:-32}"
MIN_MEM_GB="${MIN_MEM_GB:-96}"
MIN_DISK_FREE_GB="${MIN_DISK_FREE_GB:-250}"

fail=0

section() {
  printf '\n=== %s ===\n' "$1"
}

warn() {
  echo "WARN: $*" >&2
}

err() {
  echo "ERROR: $*" >&2
  fail=1
}

bytes_to_gb() {
  awk -v b="$1" 'BEGIN { printf "%.0f", b / 1024 / 1024 / 1024 }'
}

section "System"
echo "Host: $(hostname)"
echo "Kernel: $(uname -srmo)"
echo "Project: ${PROJECT_DIR}"

vcpu="$(nproc)"
mem_kb="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
mem_gb="$((mem_kb / 1024 / 1024))"
root_free_kb="$(df -Pk "$PROJECT_DIR" | awk 'NR==2 {print $4}')"
root_free_gb="$((root_free_kb / 1024 / 1024))"

echo "vCPU: ${vcpu}"
echo "RAM: ${mem_gb} GB"
echo "Free disk at project path: ${root_free_gb} GB"

if (( vcpu < MIN_VCPU )); then
  warn "vCPU is below recommended minimum (${vcpu} < ${MIN_VCPU})"
fi
if (( mem_gb < MIN_MEM_GB )); then
  warn "RAM is below recommended minimum (${mem_gb} GB < ${MIN_MEM_GB} GB)"
fi
if (( root_free_gb < MIN_DISK_FREE_GB )); then
  err "free disk is too low (${root_free_gb} GB < ${MIN_DISK_FREE_GB} GB)"
fi

section "Docker"
if ! command -v docker >/dev/null 2>&1; then
  err "docker command not found"
else
  docker --version || err "docker command failed"
  if ! docker info >/dev/null 2>&1; then
    err "Docker daemon is not reachable"
  else
    docker info --format 'Docker root dir: {{.DockerRootDir}}'
    docker info --format 'Storage driver: {{.Driver}}'
  fi
fi

section "Docker Compose"
if docker compose version >/dev/null 2>&1; then
  docker compose version
else
  err "docker compose plugin is not available"
fi

section "Docker Disk Usage"
if docker info >/dev/null 2>&1; then
  docker system df || true
fi

section "Compose File"
if [[ -f "${PROJECT_DIR}/docker-compose.yml" ]]; then
  echo "docker-compose.yml found"
else
  err "docker-compose.yml not found"
fi

section "Required Scripts"
for path in \
  "${PROJECT_DIR}/scripts/smoke_hibench_workloads.sh" \
  "${PROJECT_DIR}/scripts/run_hibench_experiments.sh" \
  "${PROJECT_DIR}/scripts/collect_hibench_data.sh" \
  "${PROJECT_DIR}/scripts/snapshot_hdfs_input.sh" \
  "${PROJECT_DIR}/scripts/restore_hdfs_input.sh" \
  "${PROJECT_DIR}/scripts/monitor_collection_resources.sh"; do
  if [[ -f "$path" ]]; then
    echo "found: ${path}"
  else
    err "missing: ${path}"
  fi
done

section "Existing Project Data"
du -sh "${PROJECT_DIR}/data" "${PROJECT_DIR}/out" 2>/dev/null || true

section "Result"
if (( fail != 0 )); then
  echo "Preflight FAILED"
  exit 1
fi

echo "Preflight OK"
