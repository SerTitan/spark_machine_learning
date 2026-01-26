#!/usr/bin/env bash
set -euo pipefail
#
# run_specific_config.sh — Запуск конкретной конфигурации из report.json
#
# Использование:
#   ./run_specific_config.sh QLearning           # Запустить конфиг от Q-Learning
#   ./run_specific_config.sh DQN                 # Запустить конфиг от DQN
#   ./run_specific_config.sh Bayesian            # Запустить конфиг от Bayesian
#   ./run_specific_config.sh PPO                 # Запустить конфиг от PPO
#   ./run_specific_config.sh default             # Запустить default конфиг
#   ./run_specific_config.sh all                 # Запустить все конфиги по очереди
#   REPORT_JSON=path/to/report.json ./run_specific_config.sh QLearning
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
REPORT_JSON="${REPORT_JSON:-${PROJECT_DIR}/out/rl_topo336/report.json}"
DOCKER_NET="${DOCKER_NET:-spark_machine_learning_bench-net}"
REPEATS="${REPEATS:-6}"

# Проверка зависимостей
if ! command -v jq &>/dev/null; then
  echo "ERROR: jq is required. Install with: sudo apt install jq"
  exit 1
fi

if [[ ! -f "$REPORT_JSON" ]]; then
  echo "ERROR: Report file not found: $REPORT_JSON"
  exit 1
fi

# Читаем топологию из report.json
TOPO_WORKERS=$(jq -r '.topology.topology_workers' "$REPORT_JSON")
TOPO_CORES=$(jq -r '.topology.topology_worker_cores' "$REPORT_JSON")
TOPO_MEM=$(jq -r '.topology.topology_worker_mem_gb' "$REPORT_JSON")
PROFILE=$(jq -r '.profile' "$REPORT_JSON")

echo "=== Configuration Runner ==="
echo "Report: $REPORT_JSON"
echo "Topology: ${TOPO_WORKERS} workers × ${TOPO_CORES} cores × ${TOPO_MEM}GB"
echo "Profile: $PROFILE"
echo ""

# Функции для работы с Docker
_kill_workers() {
  for i in {1..12}; do
    docker rm -f "spark-worker-$i" >/dev/null 2>&1 || true
  done
}

_start_workers() {
  local n="$1" c="$2" m="$3"
  echo ">>> Starting ${n} workers (${c} cores, ${m}GB each)..."
  for i in $(seq 1 "$n"); do
    docker run -d --rm --name "spark-worker-$i" --hostname "spark-worker-$i" --network "${DOCKER_NET}" \
      -e SPARK_MODE=worker -e SPARK_MASTER_URL=spark://spark-master:7077 \
      -e SPARK_WORKER_CORES="${c}" -e SPARK_WORKER_MEMORY="${m}g" bitnami/spark:3.3 >/dev/null
  done
  # Ждём готовности
  for s in {1..20}; do
    if (exec 3<>/dev/tcp/localhost/7077) >/dev/null 2>&1; then
      sleep 4
      break
    fi
    sleep 1
  done
  echo ">>> Workers ready."
}

# Конвертация формата параметров
_bool() { [[ "$1" == "1" ]] && echo "true" || echo "false"; }
_mb_to_g() { echo "$(( $1 / 1024 ))g"; }
_kb_to_k() { echo "${1}k"; }
_mb_to_m() { echo "${1}m"; }

# Извлечение конфигурации из JSON
extract_config() {
  local algo="$1"
  local prefix=""

  if [[ "$algo" == "default" ]]; then
    prefix=".default_config"
  else
    prefix=".results.${algo}.best_config"
  fi

  # Проверяем, существует ли конфиг
  if [[ $(jq -r "${prefix}" "$REPORT_JSON") == "null" ]]; then
    echo "ERROR: Configuration for '${algo}' not found in report.json"
    return 1
  fi

  # Извлекаем параметры
  local ecores=$(jq -r "${prefix}.executor_cores" "$REPORT_JSON")
  local einst=$(jq -r "${prefix}.executor_instances" "$REPORT_JSON")
  local dcores=$(jq -r "${prefix}.driver_cores" "$REPORT_JSON")
  local emem_mb=$(jq -r "${prefix}.executor_memory_mb" "$REPORT_JSON")
  local dmem_mb=$(jq -r "${prefix}.driver_memory_mb" "$REPORT_JSON")
  local mfrac=$(jq -r "${prefix}.memory_fraction" "$REPORT_JSON")
  local msfrac=$(jq -r "${prefix}.memory_storageFraction" "$REPORT_JSON")
  local rpc=$(jq -r "${prefix}.rpc_message_maxSize" "$REPORT_JSON")
  local sbuf_kb=$(jq -r "${prefix}.shuffle_file_buffer_kb" "$REPORT_JSON")
  local bblk_mb=$(jq -r "${prefix}.broadcast_block_mb" "$REPORT_JSON")
  local infl_mb=$(jq -r "${prefix}.maxSizeInFlight_mb" "$REPORT_JSON")
  local shcomp=$(jq -r "${prefix}.shuffle_compress" "$REPORT_JSON")
  local spcomp=$(jq -r "${prefix}.spill_compress" "$REPORT_JSON")
  local bcomp=$(jq -r "${prefix}.broadcast_compress" "$REPORT_JSON")
  local rddc=$(jq -r "${prefix}.rdd_compress" "$REPORT_JSON")
  local codec=$(jq -r "${prefix}.io_codec" "$REPORT_JSON")

  # Конвертируем форматы
  local emem=$(_mb_to_g "$emem_mb")
  local dmem=$(_mb_to_g "$dmem_mb")
  local sbuf=$(_kb_to_k "$sbuf_kb")
  local bblk=$(_mb_to_m "$bblk_mb")
  local infl=$(_mb_to_m "$infl_mb")
  local shcomp_b=$(_bool "$shcomp")
  local spcomp_b=$(_bool "$spcomp")
  local bcomp_b=$(_bool "$bcomp")
  local rddc_b=$(_bool "$rddc")

  # Экспортируем переменные
  export FIX_ECORES="$ecores"
  export FIX_EMEM="$emem"
  export FIX_EINST="$einst"
  export FIX_DCORES="$dcores"
  export FIX_DMEM="$dmem"
  export FIX_INFL="$infl"
  export FIX_SHUFFLE_COMP="$shcomp_b"
  export FIX_SPILL_COMP="$spcomp_b"
  export FIX_FILE_BUF="$sbuf"
  export FIX_BCAST_BLOCK="$bblk"
  export FIX_BCAST_COMP="$bcomp_b"
  export FIX_MEM_FRAC="$mfrac"
  export FIX_MEM_SFRAC="$msfrac"
  export FIX_RPC_MAX="$rpc"
  export FIX_RDD_COMP="$rddc_b"
  export FIX_CODEC="$codec"

  echo "Configuration for ${algo}:"
  echo "  executor: cores=${ecores}, memory=${emem}, instances=${einst}"
  echo "  driver: cores=${dcores}, memory=${dmem}"
  echo "  memory: fraction=${mfrac}, storageFraction=${msfrac}"
  echo "  shuffle: compress=${shcomp_b}, spill_compress=${spcomp_b}, file_buffer=${sbuf}"
  echo "  broadcast: blockSize=${bblk}, compress=${bcomp_b}"
  echo "  other: maxSizeInFlight=${infl}, rpc_maxSize=${rpc}, rdd_compress=${rddc_b}, codec=${codec}"
}

# Запуск бенчмарка с конфигурацией
run_benchmark() {
  local algo="$1"
  local output_csv="${PROJECT_DIR}/data/validation_${algo}.csv"

  echo ""
  echo "=== Running benchmark for: ${algo} ==="

  # Извлекаем конфигурацию
  extract_config "$algo" || return 1

  # Копируем скрипт в контейнер
  docker cp "${SCRIPT_DIR}/collect_wordcount_data.sh" hibench:/opt/hibench/report/collect_wordcount_data.sh
  docker exec hibench bash -c 'chmod +x /opt/hibench/report/collect_wordcount_data.sh'

  # Готовим выходной CSV
  docker exec hibench bash -c "mkdir -p /opt/hibench/report && : > /opt/hibench/report/validation_${algo}.csv"

  # Запускаем с фиксированными параметрами
  echo ">>> Running ${REPEATS} iterations..."
  docker exec -it \
    -e PROFILE="$PROFILE" \
    -e NUM_WORKERS="$TOPO_WORKERS" \
    -e WORKER_CORES="$TOPO_CORES" \
    -e WORKER_MEM_GB="$TOPO_MEM" \
    -e TARGET_SAMPLES="1" \
    -e REPEATS="$REPEATS" \
    -e USE_FIXED="1" \
    -e FIX_ECORES="$FIX_ECORES" \
    -e FIX_EMEM="$FIX_EMEM" \
    -e FIX_EINST="$FIX_EINST" \
    -e FIX_DCORES="$FIX_DCORES" \
    -e FIX_DMEM="$FIX_DMEM" \
    -e FIX_INFL="$FIX_INFL" \
    -e FIX_SHUFFLE_COMP="$FIX_SHUFFLE_COMP" \
    -e FIX_SPILL_COMP="$FIX_SPILL_COMP" \
    -e FIX_FILE_BUF="$FIX_FILE_BUF" \
    -e FIX_BCAST_BLOCK="$FIX_BCAST_BLOCK" \
    -e FIX_BCAST_COMP="$FIX_BCAST_COMP" \
    -e FIX_MEM_FRAC="$FIX_MEM_FRAC" \
    -e FIX_MEM_SFRAC="$FIX_MEM_SFRAC" \
    -e FIX_RPC_MAX="$FIX_RPC_MAX" \
    -e FIX_RDD_COMP="$FIX_RDD_COMP" \
    -e FIX_CODEC="$FIX_CODEC" \
    -e CSV="/opt/hibench/report/validation_${algo}.csv" \
    hibench bash -lc '/opt/hibench/report/collect_wordcount_data.sh'

  # Копируем результат на хост
  mkdir -p "$(dirname "$output_csv")"
  docker cp "hibench:/opt/hibench/report/validation_${algo}.csv" "$output_csv"

  # Показываем результат
  echo ""
  echo "=== Results for ${algo} ==="
  cat "$output_csv"
  echo ""

  # Извлекаем медиану
  local median=$(tail -n 1 "$output_csv" | cut -d',' -f21)
  local predicted=$(jq -r ".results.${algo}.best_predicted_time // .baseline_time_pred" "$REPORT_JSON")

  echo "Predicted time: ${predicted}s"
  echo "Actual median:  ${median}s"
  if [[ "$median" != "NA" && "$predicted" != "null" ]]; then
    local diff=$(python3 -c "print(f'{abs(float(${median}) - float(${predicted})):.2f}')")
    local pct=$(python3 -c "print(f'{100 * abs(float(${median}) - float(${predicted})) / float(${predicted}):.1f}')")
    echo "Difference:     ${diff}s (${pct}%)"
  fi
}

# Показать доступные конфигурации
show_available() {
  echo "Available configurations in report.json:"
  echo "  - default"
  for algo in $(jq -r '.results | keys[]' "$REPORT_JSON"); do
    local pred=$(jq -r ".results.${algo}.best_predicted_time" "$REPORT_JSON")
    local speedup=$(jq -r ".results.${algo}.speedup" "$REPORT_JSON")
    echo "  - ${algo} (predicted: ${pred}s, speedup: ${speedup}x)"
  done
}

# Основная логика
main() {
  if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <algorithm|all>"
    echo ""
    show_available
    exit 1
  fi

  local algo="$1"

  # Подготовка топологии
  _kill_workers
  _start_workers "$TOPO_WORKERS" "$TOPO_CORES" "$TOPO_MEM"

  # Подготовка данных (однократно)
  echo ">>> Preparing Wordcount input (profile=$PROFILE)..."
  docker exec -it hibench bash -lc "sed -i 's/^hibench.scale.profile.*/hibench.scale.profile        ${PROFILE}/' conf/hibench.conf && bin/workloads/micro/wordcount/prepare/prepare.sh || true"

  if [[ "$algo" == "all" ]]; then
    # Запускаем все конфигурации
    run_benchmark "default"
    for a in $(jq -r '.results | keys[]' "$REPORT_JSON"); do
      run_benchmark "$a"
    done

    echo ""
    echo "=== SUMMARY ==="
    echo "Results saved to:"
    echo "  - ${PROJECT_DIR}/data/validation_default.csv"
    for a in $(jq -r '.results | keys[]' "$REPORT_JSON"); do
      echo "  - ${PROJECT_DIR}/data/validation_${a}.csv"
    done
  else
    run_benchmark "$algo"
  fi

  _kill_workers
  echo ""
  echo ">>> Done!"
}

main "$@"
