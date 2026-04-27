#!/usr/bin/env bash
set -euo pipefail
#
# collect_hibench_data.sh — generic-коллектор HiBench-нагрузок.
#
# Запускается ВНУТРИ контейнера hibench. Получает на вход тип задачи (JOB_TYPE)
# и путь к воркладу (WORKLOAD_DIR), генерирует случайные валидные Spark-конфиги
# в рамках Sensors22 Table 1 (16 параметров), запускает HiBench REPEATS раз
# подряд для каждой конфигурации и пишет CSV со схемой schema_version=2.
#
# Ключевые особенности (расширение по сравнению с прошлой WordCount-версией):
#   - параметр JOB_TYPE: позволяет переиспользовать коллектор для TeraSort,
#     PageRank, WordCount (и опционально KMeans);
#   - input_size_bytes / input_dataset_id пишутся в каждую строку — модель
#     получает явный признак размера данных вместо строкового профиля;
#   - skip prepare если HDFS-вход уже существует (экономит минуты на каждой
#     ячейке матрицы сбора);
#   - self-heal hibench.conf: добавляет недостающие ключи (datatool.dir и др.),
#     иначе PageRank/KMeans падают на этапе datagen;
#   - REPEATS=3 + median + cv_duration: шумные строки отсекаются на этапе
#     обучения по cv > 0.25;
#   - MAX_TOTAL_EXECUTOR_CORES / MAX_TOTAL_EXECUTOR_MEMORY_GB ограничивают
#     суммарные ресурсы исполнителей под физический потолок 64 vCPU / 128 GB.
#
# Пример вызова из run_hibench_experiments.sh:
#   JOB_TYPE=terasort WORKLOAD_DIR=/opt/hibench/bin/workloads/micro/terasort \
#     PROFILE=large TARGET_SAMPLES=80 REPEATS=3 \
#     NUM_WORKERS=4 WORKER_CORES=8 WORKER_MEM_GB=16 \
#     bash collect_hibench_data.sh
#

cd /opt/hibench

JOB_TYPE="${JOB_TYPE:?JOB_TYPE is required, e.g. kmeans|terasort|wordcount}"
WORKLOAD_DIR="${WORKLOAD_DIR:?WORKLOAD_DIR is required, e.g. /opt/hibench/bin/workloads/micro/wordcount}"
PROFILE="${PROFILE:-large}"
CSV="${CSV:-/opt/hibench/report/hibench_train_all.csv}"
CONF="/opt/hibench/conf/spark.conf"
REPORT="/opt/hibench/report/hibench.report"

NUM_WORKERS="${NUM_WORKERS:-4}"
WORKER_CORES="${WORKER_CORES:-4}"
WORKER_MEM_GB="${WORKER_MEM_GB:-8}"

TARGET_SAMPLES="${TARGET_SAMPLES:-80}"
REPEATS="${REPEATS:-3}"
PREPARE_INPUT="${PREPARE_INPUT:-1}"
INPUT_DATASET_ID="${INPUT_DATASET_ID:-${JOB_TYPE}_${PROFILE}_default}"
INPUT_HDFS_PATH="${INPUT_HDFS_PATH:-unknown}"
INPUT_SIZE_BYTES="${INPUT_SIZE_BYTES:-0}"
MAX_TOTAL_EXECUTOR_CORES="${MAX_TOTAL_EXECUTOR_CORES:-48}"
MAX_TOTAL_EXECUTOR_MEMORY_GB="${MAX_TOTAL_EXECUTOR_MEMORY_GB:-90}"
RNG_SEED="${RNG_SEED:-20260424}"
BENCH_LOG_TAIL_LINES="${BENCH_LOG_TAIL_LINES:-200}"
SAVE_SPARK_CONF_ARTIFACTS="${SAVE_SPARK_CONF_ARTIFACTS:-0}"
SAVE_BENCH_LOG_TAILS="${SAVE_BENCH_LOG_TAILS:-0}"

SCHEMA_VERSION="${SCHEMA_VERSION:-2}"
TOTAL_CORES="$((NUM_WORKERS * WORKER_CORES))"
TOTAL_MEMORY_GB="$((NUM_WORKERS * WORKER_MEM_GB))"
RANDOM="$RNG_SEED"
ARTIFACT_ROOT="/opt/hibench/report/collection_artifacts"
CONFIG_ARTIFACT_DIR="${ARTIFACT_ROOT}/spark_conf"
TAIL_ARTIFACT_DIR="${ARTIFACT_ROOT}/bench_log_tail"
mkdir -p "$CONFIG_ARTIFACT_DIR" "$TAIL_ARTIFACT_DIR"

if [[ ! -d "$WORKLOAD_DIR" ]]; then
  echo "ERROR: WORKLOAD_DIR does not exist: $WORKLOAD_DIR" >&2
  exit 1
fi
if [[ ! -x "$WORKLOAD_DIR/spark/run.sh" ]]; then
  echo "ERROR: Spark run.sh is not executable: $WORKLOAD_DIR/spark/run.sh" >&2
  exit 1
fi

grep -q '^hibench.spark.master' "$CONF" || echo 'hibench.spark.master     spark://spark-master:7077' >> "$CONF"
grep -q '^hibench.spark.home' "$CONF" || echo 'hibench.spark.home       /opt/spark' >> "$CONF"

# Self-heal hibench.conf: the stock image misses several keys required by
# PageRank / KMeans / Bayes / SQL data generators (hibench.hibench.datatool.dir,
# hibench.hdfs.data.dir, workload dir names). We append what's missing.
HIBENCH_CONF="/opt/hibench/conf/hibench.conf"
ensure_kv() {
  local key="$1" value="$2"
  if ! grep -qE "^${key}([[:space:]]|=)" "$HIBENCH_CONF"; then
    printf '%s %s\n' "$key" "$value" >> "$HIBENCH_CONF"
  fi
}
ensure_kv "hibench.home"                 "/opt/hibench"
ensure_kv "hibench.hibench.datatool.dir" "/opt/hibench/autogen/target/autogen-8.0-SNAPSHOT-jar-with-dependencies.jar"
# hibench.hdfs.data.dir намеренно НЕ задаем: оставляем HiBench default, чтобы
# сохранить совместимость с существующими путями /Wordcount/Input, /Pagerank/Input и т.д.
ensure_kv "sparkbench.inputformat"       "Sequence"
ensure_kv "sparkbench.outputformat"      "Sequence"
ensure_kv "hibench.workload.dir.name.input"  "Input"
ensure_kv "hibench.workload.dir.name.output" "Output"
ensure_kv "hibench.masters.hostnames"    "spark-master"
ensure_kv "hibench.slaves.hostnames"     "spark-master"

if grep -qE '^hibench.scale.profile' "$HIBENCH_CONF"; then
  sed -i "s/^hibench.scale.profile.*/hibench.scale.profile        ${PROFILE}/" "$HIBENCH_CONF"
else
  echo "hibench.scale.profile        ${PROFILE}" >> "$HIBENCH_CONF"
fi

if [[ "$PREPARE_INPUT" == "1" ]]; then
  if [[ ! -x "$WORKLOAD_DIR/prepare/prepare.sh" ]]; then
    echo "ERROR: prepare.sh is not executable: $WORKLOAD_DIR/prepare/prepare.sh" >&2
    exit 1
  fi
  # Skip prepare if the expected HDFS input already exists for this profile.
  # Scripts like pagerank/kmeans use MapReduce for datagen which is slow, so
  # avoiding redundant prepare shaves minutes off each run.
  if [[ "${FORCE_PREPARE:-0}" != "1" && "$INPUT_HDFS_PATH" != "unknown" ]] && \
      /opt/hadoop/bin/hdfs dfs -test -e "$INPUT_HDFS_PATH" >/dev/null 2>&1; then
    echo ">>> [prepare] skipped: HDFS input already exists at ${INPUT_HDFS_PATH}"
  else
    echo ">>> [prepare] JOB_TYPE=${JOB_TYPE}; PROFILE=${PROFILE}; WORKLOAD_DIR=${WORKLOAD_DIR}"
    "$WORKLOAD_DIR/prepare/prepare.sh"
    echo ">>> [prepare] done."
  fi
fi

if [[ "$INPUT_SIZE_BYTES" == "0" || "$INPUT_SIZE_BYTES" == "auto" ]]; then
  if [[ "$INPUT_HDFS_PATH" != "unknown" ]] && /opt/hadoop/bin/hdfs dfs -test -e "$INPUT_HDFS_PATH" >/dev/null 2>&1; then
    INPUT_SIZE_BYTES="$(/opt/hadoop/bin/hdfs dfs -du -s "$INPUT_HDFS_PATH" 2>/dev/null | awk '{print $1}' || echo 0)"
  else
    INPUT_SIZE_BYTES="0"
  fi
fi

mkdir -p "$(dirname "$CSV")"
if [[ ! -s "$CSV" ]]; then
  {
    printf "schema_version,experiment_id,timestamp,job_type,profile,input_dataset_id,input_hdfs_path,input_size_bytes,rng_seed,"
    printf "topology_workers,topology_worker_cores,topology_worker_mem_gb,total_cores,total_memory_gb,"
    printf "executor_cores,executor_memory,executor_instances,driver_cores,driver_memory,"
    printf "memory_fraction,memory_storageFraction,shuffle_compress,spill_compress,shuffle_file_buffer,"
    printf "broadcast_block,broadcast_compress,maxSizeInFlight,io_codec,rpc_message_maxSize,rdd_compress,"
    printf "run_durations_s,median_duration_s,mean_duration_s,std_duration_s,min_duration_s,max_duration_s,cv_duration,successful_runs,exit_code,error_type,spark_conf_hash,spark_conf_path,bench_log_tail_path\n"
  } > "$CSV"
fi

_seq_int() { awk -v s="$1" -v e="$2" -v st="${3:-1}" 'BEGIN{for(i=s;i<=e;i+=st)print i}'; }
_seq_float() {
  python3 - "$@" <<'PY'
import sys
s=float(sys.argv[1]); e=float(sys.argv[2]); st=float(sys.argv[3])
x=s
out=[]
while x <= e + 1e-9:
    out.append(f"{x:.1f}")
    x += st
print("\n".join(out))
PY
}

if [[ "${USE_FIXED:-0}" == "1" ]]; then
  GRID_EXEC_CORES=("${FIX_ECORES}")
  GRID_EXEC_MEM=("${FIX_EMEM}")
  GRID_EXEC_INST=("${FIX_EINST}")
  GRID_DRIVERS=("${FIX_DCORES}")
  GRID_DRIVER_MEM=("${FIX_DMEM}")
  GRID_INFLIGHT=("${FIX_INFL}")
  GRID_SHUFFLE_COMP=("${FIX_SHUFFLE_COMP}")
  GRID_SPILL_COMP=("${FIX_SPILL_COMP}")
  GRID_FILE_BUF=("${FIX_FILE_BUF}")
  GRID_BCAST_BLOCK=("${FIX_BCAST_BLOCK}")
  GRID_BCAST_COMP=("${FIX_BCAST_COMP}")
  GRID_MEM_FRAC=("${FIX_MEM_FRAC}")
  GRID_MEM_SFRAC=("${FIX_MEM_SFRAC}")
  GRID_RPC_MAX=("${FIX_RPC_MAX}")
  GRID_RDD_COMP=("${FIX_RDD_COMP}")
  GRID_CODEC=("${FIX_CODEC}")
else
  mapfile -t GRID_EXEC_CORES < <(_seq_int 1 "$((${WORKER_CORES}<8?${WORKER_CORES}:8))" 1)
  _emax="$((${WORKER_MEM_GB}<8?${WORKER_MEM_GB}:8))"
  GRID_EXEC_MEM=()
  for g in $(_seq_int 1 "${_emax}" 1); do GRID_EXEC_MEM+=("${g}g"); done
  _imax=$((NUM_WORKERS<8?NUM_WORKERS:8))
  [[ $_imax -lt 1 ]] && _imax=1
  mapfile -t GRID_EXEC_INST < <(_seq_int 1 "$_imax" 1)
  mapfile -t GRID_DRIVERS < <(_seq_int 1 4 1)
  GRID_DRIVER_MEM=(1g 2g 3g 4g)
  GRID_INFLIGHT=(48m 56m 64m 72m 80m 88m 96m)
  GRID_SHUFFLE_COMP=(true false)
  GRID_SPILL_COMP=(true false)
  GRID_FILE_BUF=(32k 48k 64k 80k 96k 112k 128k)
  GRID_BCAST_BLOCK=(4m 6m 8m 10m 12m 14m 16m 18m 20m 22m 24m)
  GRID_BCAST_COMP=(true false)
  mapfile -t GRID_MEM_FRAC < <(_seq_float 0.3 0.8 0.1)
  mapfile -t GRID_MEM_SFRAC < <(_seq_float 0.3 0.8 0.1)
  mapfile -t GRID_RPC_MAX < <(_seq_int 128 256 32)
  GRID_RDD_COMP=(true false)
  GRID_CODEC=(lz4 snappy)
fi

_pick() { local arr=("$@"); echo "${arr[RANDOM%${#arr[@]}]}"; }
_mem_to_gb_int() {
  local s="${1,,}"
  s="${s// /}"
  if [[ "$s" == *g ]]; then
    echo "${s%g}"
  elif [[ "$s" == *m ]]; then
    echo "$(( (${s%m} + 1023) / 1024 ))"
  else
    echo "$s"
  fi
}

write_conf() {
  local ecores="$1" emem="$2" einst="$3"
  local dcores="$4" dmem="$5"
  local infl="$6"
  local shcomp="$7" spcomp="$8" sbuf="$9"
  local bblk="${10}" bcomp="${11}"
  local mfrac="${12}" msfrac="${13}"
  local rpcmax="${14}"
  local rddc="${15}"
  local codec="${16}"

  cat > "$CONF" <<EOF
# --- AUTOGENERATED (collect_hibench_data.sh) ---
hibench.spark.master     spark://spark-master:7077
hibench.spark.home       /opt/spark

spark.executor.cores     ${ecores}
spark.executor.memory    ${emem}
spark.executor.instances ${einst}

spark.driver.cores       ${dcores}
spark.driver.memory      ${dmem}

spark.reducer.maxSizeInFlight    ${infl}
spark.shuffle.compress           ${shcomp}
spark.shuffle.spill.compress     ${spcomp}
spark.shuffle.file.buffer        ${sbuf}
spark.broadcast.blockSize        ${bblk}
spark.broadcast.compress         ${bcomp}
spark.memory.fraction            ${mfrac}
spark.memory.storageFraction     ${msfrac}
spark.rpc.message.maxSize        ${rpcmax}
spark.rdd.compress               ${rddc}
spark.io.compression.codec       ${codec}

spark.serializer         org.apache.spark.serializer.KryoSerializer
spark.eventLog.enabled   true
spark.eventLog.dir       /opt/spark/history
EOF
}

run_once() {
  "$WORKLOAD_DIR/spark/run.sh"
}

bench_log_path() {
  local phase="$1"
  local candidate="/opt/hibench/report/${JOB_TYPE}/${phase}/bench.log"
  if [[ -f "$candidate" ]]; then
    echo "$candidate"
    return
  fi
  find "/opt/hibench/report/${JOB_TYPE}" -path "*/${phase}/bench.log" -type f 2>/dev/null | head -n 1
}

save_bench_tail() {
  local phase="$1" out="$2"
  local log_path
  log_path="$(bench_log_path "$phase")"
  if [[ -n "$log_path" && -f "$log_path" ]]; then
    tail -n "$BENCH_LOG_TAIL_LINES" "$log_path" > "$out"
  else
    printf "bench.log not found for job_type=%s phase=%s\n" "$JOB_TYPE" "$phase" > "$out"
  fi
}

stats_json() {
  python3 - "$@" <<'PY'
import json
import statistics
import sys

vals = [float(x) for x in sys.argv[1:] if x and x != "NA"]
if not vals:
    print(json.dumps({"median": "NA", "mean": "NA", "std": "NA", "min": "NA", "max": "NA", "cv": "NA"}))
    sys.exit(0)
mean = statistics.mean(vals)
std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
cv = std / mean if mean else 0.0
print(json.dumps({
    "median": statistics.median(vals),
    "mean": mean,
    "std": std,
    "min": min(vals),
    "max": max(vals),
    "cv": cv,
}))
PY
}

csv_quote() {
  local s="${1//\"/\"\"}"
  printf '"%s"' "$s"
}

echo "=== START collect: JOB_TYPE=${JOB_TYPE}; PROFILE=${PROFILE}; TARGET_SAMPLES=${TARGET_SAMPLES}; REPEATS=${REPEATS}; TOPOLOGY=${NUM_WORKERS}x${WORKER_CORES}c${WORKER_MEM_GB}g ==="

samples_done=0
while [[ "$samples_done" -lt "$TARGET_SAMPLES" ]]; do
  for attempt in $(seq 1 1000); do
    ecores="$(_pick "${GRID_EXEC_CORES[@]}")"
    emem="$(_pick "${GRID_EXEC_MEM[@]}")"
    einst="$(_pick "${GRID_EXEC_INST[@]}")"
    dcores="$(_pick "${GRID_DRIVERS[@]}")"
    dmem="$(_pick "${GRID_DRIVER_MEM[@]}")"
    infl="$(_pick "${GRID_INFLIGHT[@]}")"
    shcomp="$(_pick "${GRID_SHUFFLE_COMP[@]}")"
    spcomp="$(_pick "${GRID_SPILL_COMP[@]}")"
    sbuf="$(_pick "${GRID_FILE_BUF[@]}")"
    bblk="$(_pick "${GRID_BCAST_BLOCK[@]}")"
    bcomp="$(_pick "${GRID_BCAST_COMP[@]}")"
    mfrac="$(_pick "${GRID_MEM_FRAC[@]}")"
    msfrac="$(_pick "${GRID_MEM_SFRAC[@]}")"
    rpcmax="$(_pick "${GRID_RPC_MAX[@]}")"
    rddc="$(_pick "${GRID_RDD_COMP[@]}")"
    codec="$(_pick "${GRID_CODEC[@]}")"

    emem_gb="$(_mem_to_gb_int "$emem")"
    total_executor_cores=$((ecores * einst))
    total_executor_memory_gb=$((emem_gb * einst))
    if (( total_executor_cores <= MAX_TOTAL_EXECUTOR_CORES && total_executor_memory_gb <= MAX_TOTAL_EXECUTOR_MEMORY_GB )); then
      break
    fi
    if [[ "$attempt" -eq 1000 ]]; then
      echo "ERROR: failed to sample a valid Spark config under current constraints" >&2
      exit 1
    fi
  done

  experiment_id="${JOB_TYPE}_${PROFILE}_${NUM_WORKERS}x${WORKER_CORES}x${WORKER_MEM_GB}_$(date +%Y%m%d%H%M%S)_$((samples_done+1))"
  echo ">>> [$((samples_done+1))/$TARGET_SAMPLES] ${experiment_id}: e.cores=$ecores e.mem=$emem e.inst=$einst d.cores=$dcores d.mem=$dmem"

  write_conf "$ecores" "$emem" "$einst" "$dcores" "$dmem" "$infl" "$shcomp" "$spcomp" "$sbuf" "$bblk" "$bcomp" "$mfrac" "$msfrac" "$rpcmax" "$rddc" "$codec"
  spark_conf_hash="$(sha256sum "$CONF" | awk '{print $1}')"
  spark_conf_path=""
  if [[ "$SAVE_SPARK_CONF_ARTIFACTS" == "1" ]]; then
    spark_conf_path="${CONFIG_ARTIFACT_DIR}/${experiment_id}.spark.conf"
    cp "$CONF" "$spark_conf_path"
  fi

  ok=0
  durations=()
  tail_paths=()
  for r in $(seq 1 "$REPEATS"); do
    before_lines=$(wc -l < "$REPORT" 2>/dev/null || echo 0)
    set +e
    run_once
    rc=$?
    set -e
    if [[ "$SAVE_BENCH_LOG_TAILS" == "1" ]]; then
      tail_path="${TAIL_ARTIFACT_DIR}/${experiment_id}_run${r}.bench.tail.log"
      save_bench_tail "spark" "$tail_path"
      tail_paths+=("$tail_path")
    fi
    after_lines=$(wc -l < "$REPORT" 2>/dev/null || echo 0)
    if [[ "$rc" -eq 0 && "$after_lines" -gt "$before_lines" ]]; then
      duration="$(tail -n 1 "$REPORT" | awk '{print $5}')"
      durations+=("$duration")
      ok=$((ok+1))
      echo "    - run #$r / $REPEATS: ${duration}s"
    else
      echo "    - run #$r / $REPEATS failed: rc=${rc}"
    fi
  done

  stats="$(stats_json "${durations[@]}")"
  median="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["median"])' <<< "$stats")"
  mean="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["mean"])' <<< "$stats")"
  std="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["std"])' <<< "$stats")"
  min_v="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["min"])' <<< "$stats")"
  max_v="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["max"])' <<< "$stats")"
  cv="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["cv"])' <<< "$stats")"

  rc_all=$([[ "$ok" -eq "$REPEATS" ]] && echo 0 || echo 1)
  if [[ "$ok" -eq "$REPEATS" ]]; then
    error_type="ok"
  elif [[ "$ok" -gt 0 ]]; then
    error_type="partial_run_failed"
  else
    error_type="run_failed"
  fi
  run_durations="$(IFS=';'; echo "${durations[*]:-}")"
  bench_log_tail_path="$(IFS=';'; echo "${tail_paths[*]:-}")"

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$SCHEMA_VERSION" \
    "$experiment_id" \
    "$(date --iso-8601=seconds)" \
    "$JOB_TYPE" \
    "$PROFILE" \
    "$INPUT_DATASET_ID" \
    "$(csv_quote "$INPUT_HDFS_PATH")" \
    "$INPUT_SIZE_BYTES" \
    "$RNG_SEED" \
    "$NUM_WORKERS" \
    "$WORKER_CORES" \
    "$WORKER_MEM_GB" \
    "$TOTAL_CORES" \
    "$TOTAL_MEMORY_GB" \
    "$ecores" \
    "$emem" \
    "$einst" \
    "$dcores" \
    "$dmem" \
    "$mfrac" \
    "$msfrac" \
    "$shcomp" \
    "$spcomp" \
    "$sbuf" \
    "$bblk" \
    "$bcomp" \
    "$infl" \
    "$codec" \
    "$rpcmax" \
    "$rddc" \
    "$(csv_quote "$run_durations")" \
    "$median" \
    "$mean" \
    "$std" \
    "$min_v" \
    "$max_v" \
    "$cv" \
    "$ok" \
    "$rc_all" \
    "$error_type" \
    "$spark_conf_hash" \
    "$(csv_quote "$spark_conf_path")" \
    "$(csv_quote "$bench_log_tail_path")" >> "$CSV"

  echo "    -> median=${median}s mean=${mean}s std=${std}s ok=${ok}/${REPEATS} error_type=${error_type}"
  samples_done=$((samples_done+1))
done

echo "=== DONE collect: wrote ${TARGET_SAMPLES} rows to ${CSV} ==="
