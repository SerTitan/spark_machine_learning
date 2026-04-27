# Training Experiments

Последнее обновление: 2026-04-27.

## Текущая модель для API

Runtime API загружает модели из `out/final_best/<job_type>/`. Сейчас поддерживается только `job_type=pagerank`, поэтому актуальная production-модель лежит в `out/final_best/pagerank/`.

`ModelRegistry` выбирает лучший доступный `model_*.joblib` по минимальному `MAE` из `report.json`. Для PageRank это `RandomForest_RandomSearch`:

| Dataset | Rows | Лучшая модель | MAE (s) | RMSE (s) | R² | MAPE (%) |
|---------|------|---------------|---------|----------|----|----------|
| `data/hibench_train_20260424_175032_clean.csv` | 917 | RandomForest_RandomSearch | **8.972** | 16.509 | 0.844 | 15.78 |

Команда запуска:

```bash
.venv/bin/python training/train_baseline.py \
  --csv data/hibench_train_20260424_175032_clean.csv \
  --outdir out/final_best/pagerank \
  --rf-search-iters 200 \
  --sa-iters 250 \
  --mlp-hidden 64,32 \
  --mlp-lr 0.0025 \
  --mlp-max-iter 1400 \
  --mlp-patience 60 \
  --seed 42
```

## Предыдущий WordCount baseline

Dataset: `data/wc_train_merged.csv` (401 rows, profile=large). Splits: train 64%, val 16%, test 20% (stratified where possible). All runs use the updated preprocessing in `models/data.py`.

## Итоговые запуски (test set)
| Run 				  | Лучшая модель   				| MAE (s) 	| RMSE (s) | R²    | MAPE (%) |
|---------------------|---------------------------------|-----------|----------|-------|----------|
| final_best_baseline | RandomForest_SimulatedAnnealing | **0.746** | 1.215    | 0.961 | 3.19 	  |
| final_best_dnn      | DNN 							| 3.038 	| 3.949    | 0.591 | 13.18    |

Команды запуска:
- Baseline: `train_baseline.py --csv data/wc_train_merged.csv --outdir out/final_best/baseline --skip-dummy --rf-search-iters 200 --sa-iters 250 --mlp-hidden 64,32 --mlp-lr 0.0025 --mlp-max-iter 1400 --mlp-patience 60 --seed 42`
- DNN: `train_dnn.py --csv data/wc_train_merged.csv --outdir out/final_best/dnn --hidden-sizes 64,32,16 --dropout 0 --lr 0.003 --batch-size 16 --epochs 500 --patience 80 --seed 42`

## Артефакты
- PageRank API: `out/final_best/pagerank/report.json`, `preprocessor.joblib`, лучший `model_randomforest_randomsearch.joblib`, графики в `out/final_best/pagerank/plots/`.
- WordCount baseline: `out/final_best/baseline/`.
- DNN WordCount: `out/final_best/dnn/` оставлен как эксперимент, но не используется в API.
- Общие графики WordCount baseline + DNN: `out/final_best/plots/panel_metrics.png`, `scatter_all.png`. Можно пересобрать: `python scripts/plot_combined.py --baseline-dir out/final_best/baseline --dnn-dir out/final_best/dnn --csv data/wc_train_merged.csv --seed 42 --outdir out/final_best/plots`

## Что коммитить из `out`

Коммитить стоит только воспроизводимые финальные артефакты, которые нужны для запуска API и отчёта:

- `out/final_best/pagerank/report.json`
- `out/final_best/pagerank/metrics_baseline.csv`
- `out/final_best/pagerank/preprocessor.joblib`
- `out/final_best/pagerank/model_randomforest_randomsearch.joblib`
- `out/final_best/pagerank/plots/*.png`
- уже существующие WordCount/DNN/RL артефакты можно оставить, если они нужны для текста НИР/ВКР.

Не коммитить дубликаты и локальный мусор:

- `out/final_best/baseline/pagerank/` — удалён как дубль `out/final_best/pagerank/`
- `.coverage`, `.pytest_cache/`, `__pycache__/`
- локальные SQLite-файлы истории, если они не нужны как демонстрационные данные

## Краткие выводы по графикам
- MAE / RMSE: RF_SA и RF_RandomSearch существенно опережают остальные; DNN заметно хуже деревьев, но лучше Dummy (который отключён).
- R²: RF-модели ≈0.96, ExtraTrees ≈0.95; DNN ~0.59 — объясняет меньше дисперсии из-за малого датасета и отсутствия CUDA-ускорения.
- MAPE: у RF_SA ~3%, у DNN ~13% — ошибки DNN по отношению к истинным значениям в 4–5 раз выше.
- Scatter RF_SA vs DNN: точки RF_SA ближе к диагонали, у DNN — более широкий разброс, особенно для больших длительностей.
