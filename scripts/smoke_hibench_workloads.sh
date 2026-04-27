#!/usr/bin/env bash
set -euo pipefail
#
# smoke_hibench_workloads.sh — быстрая проверка, что все нагрузки запускаются.
#
# До основного сбора проверяет на одной фиксированной топологии,
# что все WORKLOADS из списка проходят prepare + run без ошибок,
# и грубо измеряет среднее время прогона. По этим временам
# принимается решение о размере матрицы основного сбора.
#
# Отдельно лечит две типовые проблемы образа sertitanius/spark_machine_learning-hibench:
#   1) пропущенный hibench.hibench.datatool.dir → datagen падает;
#   2) непрожевленные YARN NodeManager-ы → prepare PageRank/KMeans висит в ACCEPTED.
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
DATA_DIR="${PROJECT_DIR}/data"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

DOCKER_NET="${DOCKER_NET:-spark_machine_learning_bench-net}"
SPARK_WORKER_IMAGE="${SPARK_WORKER_IMAGE:-sertitanius/spark_machine_learning-spark:3.3}"
WORKLOADS="${WORKLOADS:-kmeans,terasort,wordcount}"
PROFILES="${PROFILES:-small,large,huge}"
REPEATS="${REPEATS:-1}"
START_WORKERS="${START_WORKERS:-1}"
TOPO_WORKERS="${TOPO_WORKERS:-4}"
TOPO_CORES="${TOPO_CORES:-4}"
TOPO_MEM_GB="${TOPO_MEM_GB:-8}"
OUT_CSV="${OUT_CSV:-${DATA_DIR}/smoke_hibench_workloads_${TIMESTAMP}.csv}"
WORKER_CONTAINER_CPUS="${WORKER_CONTAINER_CPUS:-${TOPO_CORES}}"
WORKER_CONTAINER_MEM_GB="${WORKER_CONTAINER_MEM_GB:-${TOPO_MEM_GB}}"

mkdir -p "$DATA_DIR"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $1" >&2
    exit 1
  fi
}

need_cmd docker

docker exec hibench bash -lc 'test -d /opt/hibench && test -d /opt/hibench/bin/workloads' >/dev/null

# Self-heal hibench.conf: stock образ пропускает hibench.hibench.datatool.dir,
# из-за чего падают prepare PageRank/KMeans/Bayes/SQL. Добавляем недостающие ключи.
docker exec hibench bash -lc '
HIBENCH_CONF="/opt/hibench/conf/hibench.conf"
ensure_kv() {
  local key="$1" value="$2"
  if ! grep -qE "^${key}([[:space:]]|=)" "$HIBENCH_CONF"; then
    printf "%s %s\n" "$key" "$value" >> "$HIBENCH_CONF"
  fi
}
ensure_kv "hibench.home"                 "/opt/hibench"
ensure_kv "hibench.hibench.datatool.dir" "/opt/hibench/autogen/target/autogen-8.0-SNAPSHOT-jar-with-dependencies.jar"
ensure_kv "sparkbench.inputformat"       "Sequence"
ensure_kv "sparkbench.outputformat"      "Sequence"
ensure_kv "hibench.workload.dir.name.input"  "Input"
ensure_kv "hibench.workload.dir.name.output" "Output"
ensure_kv "hibench.masters.hostnames"    "spark-master"
ensure_kv "hibench.slaves.hostnames"     "spark-master"
'

# PageRank/KMeans/TeraSort используют MapReduce для prepare. Без YARN NodeManager-ов
# они виснут в очереди. Проверяем, что хотя бы один NM зарегистрирован.
NM_COUNT="$(docker exec resourcemanager bash -lc "/opt/hadoop/bin/yarn node -list 2>/dev/null | grep -c RUNNING" || echo 0)"
if [[ "${NM_COUNT}" -lt 1 ]]; then
  echo "ERROR: no YARN NodeManagers registered. Start them with:" >&2
  echo "  docker compose up -d nodemanager-1 nodemanager-2" >&2
  echo "Without NMs, PageRank/KMeans/TeraSort prepare will hang." >&2
  exit 1
fi
echo ">>> YARN NodeManagers online: ${NM_COUNT}"

kill_workers() {
  for i in $(seq 1 32); do
    docker rm -f "spark-worker-$i" >/dev/null 2>&1 || true
  done
}

start_workers() {
  local n="$1" c="$2" m="$3"
  echo ">>> Starting Spark workers: ${n} workers x ${c} cores x ${m} GB"
  kill_workers
  for i in $(seq 1 "$n"); do
    docker run -d --rm \
      --name "spark-worker-$i" \
      --hostname "spark-worker-$i" \
      --network "$DOCKER_NET" \
      --cpus "$WORKER_CONTAINER_CPUS" \
      --memory "${WORKER_CONTAINER_MEM_GB}g" \
      --memory-swap "${WORKER_CONTAINER_MEM_GB}g" \
      -e SPARK_MODE=worker \
      -e SPARK_MASTER_URL=spark://spark-master:7077 \
      -e SPARK_WORKER_CORES="$c" \
      -e SPARK_WORKER_MEMORY="${m}g" \
      "$SPARK_WORKER_IMAGE" >/dev/null
  done
  sleep 6
}

resolve_workload_dir() {
  local workload="$1"
  docker exec hibench bash -lc "python3 - <<'PY'
from pathlib import Path
import sys

name = '${workload}'.lower()
candidates = {
    'wordcount': [
        '/opt/hibench/bin/workloads/micro/wordcount',
    ],
    'pagerank': [
        '/opt/hibench/bin/workloads/websearch/pagerank',
        '/opt/hibench/bin/workloads/graph/pagerank',
    ],
    'kmeans': [
        '/opt/hibench/bin/workloads/ml/kmeans',
        '/opt/hibench/bin/workloads/machinelearning/kmeans',
    ],
    'sort': [
        '/opt/hibench/bin/workloads/micro/sort',
    ],
    'terasort': [
        '/opt/hibench/bin/workloads/micro/terasort',
    ],
}.get(name, [])

for raw in candidates:
    p = Path(raw)
    if (p / 'prepare' / 'prepare.sh').exists() and (p / 'spark' / 'run.sh').exists():
        print(p)
        sys.exit(0)

for p in Path('/opt/hibench/bin/workloads').rglob(name):
    if p.is_dir() and (p / 'prepare' / 'prepare.sh').exists() and (p / 'spark' / 'run.sh').exists():
        print(p)
        sys.exit(0)

sys.exit(1)
PY"
}

expected_input_path() {
  local workload="$1"
  case "${workload,,}" in
    wordcount) echo "/Wordcount/Input" ;;
    pagerank) echo "/Pagerank/Input" ;;
    kmeans) echo "/Kmeans/Input" ;;
    sort|terasort) echo "/Terasort/Input" ;;
    *) echo "" ;;
  esac
}

csv_escape() {
  local s="${1//\"/\"\"}"
  printf '"%s"' "$s"
}

classify_error() {
  local prepare_rc="$1" input_ok="$2" run_rc="$3"
  if [[ "$prepare_rc" -ne 0 ]]; then
    echo "prepare_failed"
  elif [[ "$input_ok" -ne 1 ]]; then
    echo "input_missing"
  elif [[ "$run_rc" -ne 0 ]]; then
    echo "run_failed"
  else
    echo "ok"
  fi
}

echo "workload,profile,repeat,workload_dir,expected_input_path,prepare_rc,run_rc,error_type,duration_s,report_lines_before,report_lines_after" > "$OUT_CSV"

if [[ "$START_WORKERS" == "1" ]]; then
  start_workers "$TOPO_WORKERS" "$TOPO_CORES" "$TOPO_MEM_GB"
fi

IFS=',' read -r -a workload_arr <<< "$WORKLOADS"
IFS=',' read -r -a profile_arr <<< "$PROFILES"

for workload in "${workload_arr[@]}"; do
  workload="$(echo "$workload" | xargs)"
  echo "=== Workload: ${workload} ==="

  set +e
  workload_dir="$(resolve_workload_dir "$workload")"
  resolve_rc=$?
  set -e

  if [[ "$resolve_rc" -ne 0 || -z "$workload_dir" ]]; then
    echo "WARN: workload not found or unsupported in this HiBench image: ${workload}"
    for profile in "${profile_arr[@]}"; do
      profile="$(echo "$profile" | xargs)"
      echo "${workload},${profile},0,,,127,127,workload_not_found,0,0,0" >> "$OUT_CSV"
    done
    continue
  fi

  echo ">>> Resolved path: ${workload_dir}"

  for profile in "${profile_arr[@]}"; do
    profile="$(echo "$profile" | xargs)"
    echo "--- Profile: ${profile} ---"

    set +e
    docker exec hibench bash -lc "cd /opt/hibench && sed -i 's/^hibench.scale.profile.*/hibench.scale.profile        ${profile}/' conf/hibench.conf && ${workload_dir}/prepare/prepare.sh"
    prepare_rc=$?
    set -e
    expected_input="$(expected_input_path "$workload")"
    input_ok=0
    if [[ -n "$expected_input" ]]; then
      if docker exec hibench bash -lc "/opt/hadoop/bin/hdfs dfs -test -e '${expected_input}'" >/dev/null 2>&1; then
        input_ok=1
      else
        echo "WARN: expected HDFS input was not created: ${expected_input}"
      fi
    else
      input_ok=1
    fi

    for repeat in $(seq 1 "$REPEATS"); do
      before_lines="$(docker exec hibench bash -lc 'test -f /opt/hibench/report/hibench.report && wc -l < /opt/hibench/report/hibench.report || echo 0' | tr -d '\r')"
      start_epoch="$(date +%s)"

      if [[ "$prepare_rc" -eq 0 && "$input_ok" -eq 1 ]]; then
        set +e
        docker exec hibench bash -lc "cd /opt/hibench && ${workload_dir}/spark/run.sh"
        run_rc=$?
        set -e
      else
        run_rc=125
      fi

      end_epoch="$(date +%s)"
      after_lines="$(docker exec hibench bash -lc 'test -f /opt/hibench/report/hibench.report && wc -l < /opt/hibench/report/hibench.report || echo 0' | tr -d '\r')"
      duration_s="$((end_epoch - start_epoch))"
      error_type="$(classify_error "$prepare_rc" "$input_ok" "$run_rc")"

      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$workload" \
        "$profile" \
        "$repeat" \
        "$(csv_escape "$workload_dir")" \
        "$(csv_escape "$expected_input")" \
        "$prepare_rc" \
        "$run_rc" \
        "$error_type" \
        "$duration_s" \
        "$before_lines" \
        "$after_lines" >> "$OUT_CSV"

      echo "repeat=${repeat}; prepare_rc=${prepare_rc}; run_rc=${run_rc}; error_type=${error_type}; duration=${duration_s}s"
    done
  done
done

if [[ "$START_WORKERS" == "1" ]]; then
  kill_workers
fi

echo ">>> Smoke CSV saved to: ${OUT_CSV}"
