# Training Experiments

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
- Общие графики (baseline + DNN, понятные подписи и шкалы): `out/final_best/plots/panel_metrics.png`, `scatter_all.png`. Можно пересобрать: `python scripts/plot_combined.py --baseline-dir out/final_best/baseline --dnn-dir out/final_best/dnn --csv data/wc_train_merged.csv --seed 42 --outdir out/final_best/plots`
- Отчёты и модели: `out/final_best/baseline/`, `out/final_best/dnn/`.

## Краткие выводы по графикам
- MAE / RMSE: RF_SA и RF_RandomSearch существенно опережают остальные; DNN заметно хуже деревьев, но лучше Dummy (который отключён).
- R²: RF-модели ≈0.96, ExtraTrees ≈0.95; DNN ~0.59 — объясняет меньше дисперсии из-за малого датасета и отсутствия CUDA-ускорения.
- MAPE: у RF_SA ~3%, у DNN ~13% — ошибки DNN по отношению к истинным значениям в 4–5 раз выше.
- Scatter RF_SA vs DNN: точки RF_SA ближе к диагонали, у DNN — более широкий разброс, особенно для больших длительностей.
