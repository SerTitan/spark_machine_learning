#!/usr/bin/env python3
import argparse, json, os
import numpy as np, pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

import matplotlib.pyplot as plt

# MLflow (assumes server at localhost:5000)
import mlflow, mlflow.sklearn
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("hibench_wordcount_baseline")

rng = np.random.RandomState(0)

def mape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (np.maximum(np.abs(y_true), eps)))) * 100

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def simulated_annealing_rf(X_train, y_train, X_val, y_val, iters=60, T0=3.0, alpha=0.93):
    def random_params():
        return {
            "n_estimators": int(rng.randint(80, 400)),
            "max_depth": int(rng.randint(3, 22)),
            "min_samples_split": int(rng.randint(2, 12)),
            "min_samples_leaf": int(rng.randint(1, 8)),
            "max_features": rng.choice(["sqrt", "log2", None])
        }
    def score(params):
        model = RandomForestRegressor(random_state=0, n_jobs=-1, **params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        return rmse(y_val, pred), model

    cur_params = random_params()
    cur_score, cur_model = score(cur_params)
    best_params, best_score, best_model = cur_params, cur_score, cur_model
    T = T0
    history = [(0, cur_score)]
    for t in range(1, iters+1):
        cand = random_params()
        cand_score, cand_model = score(cand)
        if cand_score < cur_score or rng.rand() < np.exp((cur_score - cand_score) / max(T, 1e-8)):
            cur_params, cur_score, cur_model = cand, cand_score, cand_model
        if cur_score < best_score:
            best_params, best_score, best_model = cur_params, cur_score, cur_model
        history.append((t, cur_score))
        T *= alpha
    return best_model, best_params, best_score, history

def build_dataset(df):
    df = df.copy()
    if "rdd_compress" in df.columns:
        df["rdd_compress"] = df["rdd_compress"].astype(str).str.lower().map({"true":1,"false":0}).fillna(0).astype(int)
    if "is_default" not in df.columns:
        df["is_default"] = 0
    feat = ["executors","executor_cores","executor_memory_gb",
            "driver_cores","driver_memory_gb","parallelism","shuffle_partitions",
            "rdd_compress","io_compression_codec","input_gb"]
    feat = [c for c in feat if c in df.columns]
    target_col = "median_sec" if "median_sec" in df.columns else "duration_sec"
    df = df.dropna(subset=[target_col])
    cat = [c for c in ["io_compression_codec"] if c in feat]
    num = [c for c in feat if c not in cat]
    X = df[feat]
    y = df[target_col].astype(float).values
    return X, y, num, cat, df

def plot_and_save(figpath):
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()

def main(csv_path, outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    df_in = pd.read_csv(csv_path)
    X, y, num, cat, df_full = build_dataset(df_in)

    pre = ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    results = {}

    with mlflow.start_run(run_name="baseline_large"):
        mlflow.log_params({"dataset": str(csv_path), "rows": len(df_in)})

        # Dummy
        dummy = Pipeline([("pre", pre), ("m", DummyRegressor(strategy="median"))])
        dummy.fit(X_train, y_train)
        y_pred = dummy.predict(X_test)
        results["Dummy"] = {"MAE": mean_absolute_error(y_test, y_pred),
                            "RMSE": rmse(y_test, y_pred),
                            "R2": r2_score(y_test, y_pred),
                            "MAPE": mape(y_test, y_pred)}

        # RF + Random Search
        rf = Pipeline([("pre", pre), ("m", RandomForestRegressor(random_state=0, n_jobs=-1))])
        param_space = {
            "m__n_estimators": np.arange(80, 401),
            "m__max_depth": np.arange(3, 22),
            "m__min_samples_split": np.arange(2, 12),
            "m__min_samples_leaf": np.arange(1, 8),
            "m__max_features": ["sqrt","log2", None]
        }
        rs = RandomizedSearchCV(rf, param_space, n_iter=40, cv=3, random_state=0,
                                n_jobs=-1, scoring="neg_root_mean_squared_error")
        rs.fit(X_train, y_train)
        y_pred = rs.best_estimator_.predict(X_test)
        results["RandomSearch_RF"] = {"MAE": mean_absolute_error(y_test, y_pred),
                                      "RMSE": rmse(y_test, y_pred),
                                      "R2": r2_score(y_test, y_pred),
                                      "MAPE": mape(y_test, y_pred)}

        # SA over RF (train on transformed arrays for speed)
        Xtr = rs.best_estimator_.named_steps["pre"].transform(X_train)
        Xte = rs.best_estimator_.named_steps["pre"].transform(X_test)
        best_model, best_params, best_score, sa_hist = simulated_annealing_rf(Xtr, y_train, Xte, y_test, iters=60)
        y_pred = best_model.predict(Xte)
        results["SimAnneal_RF"] = {"MAE": mean_absolute_error(y_test, y_pred),
                                   "RMSE": rmse(y_test, y_pred),
                                   "R2": r2_score(y_test, y_pred),
                                   "MAPE": mape(y_test, y_pred)}
        its, scores = zip(*sa_hist)
        plt.figure()
        plt.plot(list(its), list(scores))
        plt.xlabel("Iteration"); plt.ylabel("Validation RMSE"); plt.title("Simulated Annealing convergence (RF)")
        plot_and_save(outdir / "sa_convergence_rf.png")

        # DNN (MLP)
        dnn = Pipeline([("pre", pre), ("m", MLPRegressor(hidden_layer_sizes=(128,64),
                                                         activation="relu", solver="adam",
                                                         learning_rate_init=0.003, max_iter=800,
                                                         random_state=0, early_stopping=True, n_iter_no_change=20))])
        dnn.fit(X_train, y_train)
        y_pred = dnn.predict(X_test)
        results["DNN_MLP"] = {"MAE": mean_absolute_error(y_test, y_pred),
                              "RMSE": rmse(y_test, y_pred),
                              "R2": r2_score(y_test, y_pred),
                              "MAPE": mape(y_test, y_pred)}

        metrics_df = pd.DataFrame(results).T[["MAE","RMSE","R2","MAPE"]]
        metrics_csv = outdir / "metrics_baseline.csv"
        metrics_df.to_csv(metrics_csv, index=True)

        for metric in ["MAE","RMSE","R2","MAPE"] :
            plt.figure()
            metrics_df[metric].plot(kind="bar")
            plt.ylabel(metric); plt.title(f"{metric} by model")
            plot_and_save(outdir / f"bar_{metric.lower()}.png")

        sizes = np.linspace(0.2, 1.0, 6)
        dnn_scores = []
        for frac in sizes:
            X_tr, _, y_tr, _ = train_test_split(X_train, y_train, train_size=frac, random_state=0)
            m = Pipeline([("pre", pre), ("m", MLPRegressor(hidden_layer_sizes=(128,64),
                                                           activation="relu", solver="adam",
                                                           learning_rate_init=0.003, max_iter=600,
                                                           random_state=0, early_stopping=True, n_iter_no_change=15))])
            m.fit(X_tr, y_tr)
            pred = m.predict(X_test)
            dnn_scores.append(rmse(y_test, pred))
        plt.figure()
        plt.plot(sizes, dnn_scores, marker="o")
        plt.xlabel("Train fraction"); plt.ylabel("DNN RMSE"); plt.title("DNN convergence vs training size")
        plot_and_save(outdir / "dnn_convergence_vs_data.png")

        # Speedup vs default (if provided)
        speedup_png = None
        if "is_default" in df_full.columns and (df_full["is_default"]==1).any():
            defaults = df_full[df_full["is_default"]==1]["median_sec"]
            if len(defaults)>0:
                base = float(defaults.iloc[0])
                df_speed = df_full.copy()
                df_speed["speedup"] = base / df_speed["median_sec"].astype(float)
                df_speed = df_speed.sort_values("speedup", ascending=False).head(20)
                plt.figure()
                plt.barh(range(len(df_speed)), df_speed["speedup"])
                plt.yticks(range(len(df_speed)), [f"{r.executors}x{r.executor_cores}/{r.executor_memory_gb}g P{int(r.parallelism)} SP{int(r.shuffle_partitions)}"
                                                  for _, r in df_speed.iterrows()])
                plt.xlabel("Speedup vs default"); plt.title("Top speedups")
                plt.gca().invert_yaxis()
                speedup_png = outdir / "speedup_vs_default.png"
                plot_and_save(speedup_png)

        # Log to MLflow
        mlflow.log_artifact(str(metrics_csv))
        for p in ["bar_mae.png","bar_rmse.png","bar_r2.png","bar_mape.png",
                  "dnn_convergence_vs_data.png","sa_convergence_rf.png"]:
            fp = outdir / p
            if fp.exists(): mlflow.log_artifact(str(fp))
        if speedup_png and Path(speedup_png).exists():
            mlflow.log_artifact(str(speedup_png))

        for model_name, m in results.items():
            mlflow.log_metrics({f"{model_name}_MAE": m["MAE"],
                                f"{model_name}_RMSE": m["RMSE"],
                                f"{model_name}_R2": m["R2"],
                                f"{model_name}_MAPE": m["MAPE"]})

        print("Saved:", str(metrics_csv))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="dataset CSV (aggregated)")
    ap.add_argument("--outdir", required=True, help="output directory")
    args = ap.parse_args()
    main(args.csv, args.outdir)
