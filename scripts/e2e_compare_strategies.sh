#!/usr/bin/env bash
# Запускает e2e_validate.sh для нескольких стратегий подряд.
#
# Пример:
#   REPEATS=3 MODEL_NAMES="rf dnn ql" bash scripts/e2e_compare_strategies.sh
set -euo pipefail

MODEL_NAMES="${MODEL_NAMES:-rf dnn ql}"

for model_name in ${MODEL_NAMES}; do
  echo ""
  echo "################################################################################"
  echo "### E2E strategy: ${model_name}"
  echo "################################################################################"
  MODEL_NAME="${model_name}" bash scripts/e2e_validate.sh
done
