#!/usr/bin/env bash
# E2E сравнение трёх моделей: получает конфигурации от сервиса, запускает HiBench,
# сравнивает predicted_runtime vs actual_runtime для каждой модели.
#
# Использование:
#   bash scripts/e2e_compare_models.sh
#   REPEATS=5 WORKERS=4 CORES=2 RAM_GB=4 PROFILE=large bash scripts/e2e_compare_models.sh
#
# Требования: сервис запущен на :8001, Docker + HiBench + HDFS доступны.

set -euo pipefail

WORKERS=${WORKERS:-4}
CORES=${CORES:-2}
RAM_GB=${RAM_GB:-4}
PROFILE=${PROFILE:-large}
REPEATS=${REPEATS:-3}
BASE_URL=${BASE_URL:-http://localhost:8001}
HIBENCH_HOME=${HIBENCH_HOME:-/opt/hibench}
HADOOP_HOME=${HADOOP_HOME:-/opt/hadoop}
OUTFILE=${OUTFILE:-out/e2e_model_comparison.tsv}

MODELS=(rf dnn ql)

mkdir -p "$(dirname "$OUTFILE")"

# ── Получить конфигурацию от сервиса ──────────────────────────────────────────
get_config() {
    local model=$1
    curl -s -X POST "$BASE_URL/recommend" \
        -H "Content-Type: application/json" \
        -d "{
            \"job_type\": \"pagerank\",
            \"input\": {\"profile\": \"$PROFILE\"},
            \"topology\": {\"workers\": $WORKERS, \"worker_cores\": $CORES, \"worker_memory_gb\": $RAM_GB},
            \"preferences\": {\"return_top_k\": 1},
            \"model_name\": \"$model\"
        }"
}

# ── Применить конфиг к hibench.conf ──────────────────────────────────────────
apply_spark_config() {
    local conf_json=$1
    local hibench_conf=${HIBENCH_HOME}/conf/spark.conf

    executor_cores=$(echo "$conf_json" | python3 -c "import sys,json; c=json.load(sys.stdin)['recommendations'][0]['config']; print(c['executor_cores'])")
    executor_memory=$(echo "$conf_json" | python3 -c "import sys,json; c=json.load(sys.stdin)['recommendations'][0]['config']; print(c['executor_memory_mb'])")
    executor_instances=$(echo "$conf_json" | python3 -c "import sys,json; c=json.load(sys.stdin)['recommendations'][0]['config']; print(c['executor_instances'])")
    shuffle_compress=$(echo "$conf_json" | python3 -c "import sys,json; c=json.load(sys.stdin)['recommendations'][0]['config']; print('true' if c['shuffle_compress'] else 'false')")
    io_codec=$(echo "$conf_json" | python3 -c "import sys,json; c=json.load(sys.stdin)['recommendations'][0]['config']; print(c['io_codec'])")
    memory_fraction=$(echo "$conf_json" | python3 -c "import sys,json; c=json.load(sys.stdin)['recommendations'][0]['config']; print(c['memory_fraction'])")

    cat > "$hibench_conf" <<EOF
hibench.spark.master          yarn
hibench.yarn.executor.num     ${executor_instances}
hibench.yarn.executor.cores   ${executor_cores}
spark.executor.memory         ${executor_memory}m
spark.executor.instances      ${executor_instances}
spark.executor.cores          ${executor_cores}
spark.shuffle.compress        ${shuffle_compress}
spark.io.compression.codec    ${io_codec}
spark.memory.fraction         ${memory_fraction}
EOF
}

# ── Запустить HiBench PageRank и вернуть время из отчёта ─────────────────────
run_hibench_pagerank() {
    local report=${HIBENCH_HOME}/report/hibench.report
    # Очистить старую запись
    [ -f "$report" ] && tail -0 "$report" > /dev/null

    "${HIBENCH_HOME}/bin/workloads/websearch/pagerank/spark/run.sh" > /dev/null 2>&1

    # Последняя строка отчёта: поле 5 = duration (s)
    tail -1 "$report" | awk '{print $5}'
}

# ── Запустить HDFS-prepare если нужно ────────────────────────────────────────
ensure_hdfs_input() {
    if ! "${HADOOP_HOME}/bin/hdfs" dfs -test -d /Pagerank/Input 2>/dev/null; then
        echo "[prepare] Generating PageRank input data (HDFS)..."
        "${HIBENCH_HOME}/bin/workloads/websearch/pagerank/prepare/prepare.sh" > /dev/null 2>&1
    fi
}

# ── Заголовок вывода ──────────────────────────────────────────────────────────
echo ""
echo "E2E Model Comparison — PageRank"
echo "Топология: ${WORKERS}w × ${CORES}c × ${RAM_GB}GB  |  профиль: ${PROFILE}  |  повторов: ${REPEATS}"
echo ""
printf "%-6s %-10s %-8s %-8s\n" "MODEL" "RUN" "PRED(s)" "ACTUAL(s)"
echo "-----------------------------------"

printf "%-6s\t%-10s\t%-8s\t%-8s\n" "model" "run" "predicted_s" "actual_s" > "$OUTFILE"

ensure_hdfs_input

# ── Основной цикл по моделям ──────────────────────────────────────────────────
for model in "${MODELS[@]}"; do
    echo "--- Модель: $model ---"
    response=$(get_config "$model")
    predicted=$(echo "$response" | python3 -c \
        "import sys,json; print(json.load(sys.stdin)['recommendations'][0]['predicted_runtime_s'])")

    apply_spark_config "$response"

    for rep in $(seq 1 "$REPEATS"); do
        actual=$(run_hibench_pagerank)
        printf "%-6s %-10s %-8s %-8s\n" "$model" "$rep/$REPEATS" "$predicted" "$actual"
        printf "%s\t%s\t%s\t%s\n" "$model" "$rep" "$predicted" "$actual" >> "$OUTFILE"
    done
    echo ""
done

# ── Итоговая статистика ───────────────────────────────────────────────────────
echo "========================================"
echo "ИТОГ:"
python3 - "$OUTFILE" <<'PYEOF'
import sys, statistics
from collections import defaultdict

path = sys.argv[1]
data = defaultdict(lambda: {"pred": None, "actuals": []})

with open(path) as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 4:
            continue
        model, _, pred, actual = parts
        data[model]["pred"] = float(pred)
        data[model]["actuals"].append(float(actual))

print(f"{'Модель':<8} {'Predicted':>10} {'Mean actual':>12} {'CV%':>8} {'MAPE%':>8}")
print("-" * 52)
for model in ("rf", "dnn", "ql"):
    if model not in data:
        continue
    d = data[model]
    pred = d["pred"]
    acts = d["actuals"]
    mean_a = statistics.mean(acts)
    cv = statistics.stdev(acts) / mean_a * 100 if len(acts) > 1 else 0.0
    mape = abs(pred - mean_a) / mean_a * 100
    print(f"{model:<8} {pred:>10.1f} {mean_a:>12.1f} {cv:>8.1f} {mape:>8.1f}")
PYEOF

echo ""
echo "Полные данные → $OUTFILE"
