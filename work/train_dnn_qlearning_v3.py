
#!/usr/bin/env python3
"""
train_dnn_qlearning_v3.py
Improvements:
- Option --model {torch, mlp, rf}. Defaults to torch if available else mlp.
- Train on log1p(duration_s); predictions are expm1(...) and then clipped to >= 0.
- Report baseline metrics: predict per-profile median.
- Q-learning reward normalized: r = (t_prev - t_new) / max(t_prev, 1e-9).
- Robust OneHotEncoder for sklearn>=1.4.
"""

from __future__ import annotations
import argparse, json, math, os, random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TORCH_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    TORCH_AVAILABLE = False

def _to_mb(val: str) -> float:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return np.nan
    s = str(val).strip().lower()
    if s.endswith('g'): return float(s[:-1]) * 1024.0
    if s.endswith('m'): return float(s[:-1])
    if s.endswith('k'): return float(s[:-1]) / 1024.0
    try: return float(s)
    except: return np.nan

def _to_bool(val: str) -> int:
    s = str(val).strip().lower()
    if s in ('true','1','yes'): return 1
    if s in ('false','0','no'): return 0
    try: return 1 if int(s)!=0 else 0
    except: return 0

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "exit_code" in df.columns:
        df = df[df["exit_code"] == 0]
    df = df[pd.to_numeric(df["duration_s"], errors="coerce") > 0]
    df["duration_s"] = df["duration_s"].astype(float)
    return df

def make_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    work = df.copy()
    work["executor_memory_mb"] = work["executor_memory"].map(_to_mb)
    work["driver_memory_mb"] = work["driver_memory"].map(_to_mb)
    work["broadcast_block_mb"] = work["broadcast_block"].map(_to_mb)
    work["max_in_flight_mb"] = work["maxSizeInFlight"].map(_to_mb)
    work["shuffle_file_buffer_mb"] = work["shuffle_file_buffer"].map(_to_mb)
    work["shuffle_compress_b"] = work["shuffle_compress"].map(_to_bool)
    work["spill_compress_b"] = work["spill_compress"].map(_to_bool)
    y = work["duration_s"].astype(float)

    num_features = [
        "executor_cores", "executor_instances", "driver_cores",
        "memory_fraction",
        "executor_memory_mb", "driver_memory_mb",
        "broadcast_block_mb", "max_in_flight_mb", "shuffle_file_buffer_mb",
        "input_bytes"
    ]
    cat_features = ["profile", "io_codec"]
    bin_features = ["shuffle_compress_b", "spill_compress_b"]

    X = work[num_features + cat_features + bin_features].copy()
    for col in num_features + bin_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X["profile"] = X["profile"].astype(str)
    X["io_codec"] = X["io_codec"].astype(str)
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    y = y.loc[X.index]
    return X, y

class TorchRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 24), nn.ReLU(),
            nn.Linear(24, 12), nn.ReLU(),
            nn.Linear(12, 6), nn.ReLU(),
            nn.Linear(6, 1)  # log-duration
        )
    def forward(self, x):
        return self.net(x)

@dataclass
class TrainedModel:
    pre: ColumnTransformer
    model: Any
    mode: str  # 'torch' | 'mlp' | 'rf'
    feature_cols: List[str]
    cat_cols: List[str]
    bin_cols: List[str]
    num_cols: List[str]

    def predict_seconds(self, X_df: pd.DataFrame) -> np.ndarray:
        X_proc = self.pre.transform(X_df)
        if self.mode == "torch":
            self.model.eval()
            import torch
            with torch.no_grad():
                X_t = torch.from_numpy(X_proc.astype(np.float32))
                y_log = self.model(X_t).cpu().numpy().reshape(-1)
        else:
            y_log = self.model.predict(X_proc).reshape(-1)
        y_sec = np.expm1(y_log)
        y_sec = np.clip(y_sec, 0.0, None)
        return y_sec

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump({
            "pre": self.pre,
            "feature_cols": self.feature_cols,
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "bin_cols": self.bin_cols
        }, os.path.join(out_dir, "preprocess.joblib"))
        if self.mode == "torch":
            import torch
            torch.save(self.model.state_dict(), os.path.join(out_dir, "model.pt"))
        else:
            joblib.dump(self.model, os.path.join(out_dir, f"model_{self.mode}.joblib"))

def build_preprocessor(X: pd.DataFrame, num_cols: List[str], cat_cols: List[str], bin_cols: List[str]) -> ColumnTransformer:
    num_tf = Pipeline(steps=[("scaler", StandardScaler())])
    try:
        cat_tf = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    except TypeError:
        cat_tf = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    bin_tf = "passthrough"
    pre = ColumnTransformer(
        transformers=[("num", num_tf, num_cols),
                      ("cat", cat_tf, cat_cols),
                      ("bin", bin_tf, bin_cols)],
        remainder="drop"
    )
    pre.fit(X)
    return pre

def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
def mape(y_true, y_pred):
    yt, yp = np.array(y_true), np.array(y_pred)
    mask = yt != 0
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)

def train_torch(X_tr, y_tr_log, X_va, y_va_sec, *, epochs=200, lr=3e-3, batch=64, seed=42, verbose=True):
    import torch
    torch.manual_seed(seed)
    model = TorchRegressor(input_dim=X_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    ds_tr = TensorDataset(torch.from_numpy(X_tr.astype(np.float32)),
                          torch.from_numpy(y_tr_log.astype(np.float32)).view(-1,1))
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True)

    best_state = None; best_val = float("inf"); patience, bad = 20, 0
    val_hist = []

    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in dl_tr:
            opt.zero_grad()
            pred_log = model(xb)
            loss = loss_fn(pred_log, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            y_va_log_pred = model(torch.from_numpy(X_va.astype(np.float32))).cpu().numpy().reshape(-1)
        y_va_pred = np.expm1(y_va_log_pred); y_va_pred = np.clip(y_va_pred, 0.0, None)
        val_rmse = rmse(y_va_sec, y_va_pred)
        val_hist.append(val_rmse)
        if verbose and ep % 10 == 0:
            print(f"[torch] epoch {ep:03d}  val_RMSE(sec)={val_rmse:.4f}")
        if val_rmse < best_val - 1e-6:
            best_val = val_rmse; best_state = {k: v.clone() for k,v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"[torch] early stop at epoch {ep}, best val_RMSE={best_val:.4f}")
                break
    if best_state: model.load_state_dict(best_state)
    return model, val_hist

def train_sklearn_mlp(X_tr, y_tr_log, X_va, y_va_sec):
    from sklearn.neural_network import MLPRegressor
    mlp = MLPRegressor(hidden_layer_sizes=(24,12,6),
                       activation="relu",
                       solver="adam",
                       learning_rate_init=3e-3,
                       batch_size=64,
                       max_iter=800,
                       random_state=42,
                       early_stopping=True,
                       n_iter_no_change=40,
                       verbose=False)
    mlp.fit(X_tr, y_tr_log)
    return mlp, getattr(mlp, "loss_curve_", [])

def train_rf(X_tr, y_tr_log, X_va, y_va_sec):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr_log)  # fit on log-target
    return rf, []

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def plot_pred_vs_actual(y_true, y_pred, out_path):
    plt.figure(); plt.scatter(y_true, y_pred, s=12)
    mmin = min(min(y_true), min(y_pred)); mmax = max(max(y_true), max(y_pred))
    xs = np.linspace(mmin, mmax, 100); plt.plot(xs, xs)
    plt.xlabel("Actual duration (s)"); plt.ylabel("Predicted duration (s)")
    plt.title("Predicted vs Actual (test)"); plt.tight_layout(); plt.savefig(out_path); plt.close()
def plot_residuals(y_true, y_pred, out_path):
    res = np.array(y_true) - np.array(y_pred)
    plt.figure(); plt.hist(res, bins=30); plt.xlabel("Residual (actual - predicted)"); plt.ylabel("Count")
    plt.title("Residuals (test)"); plt.tight_layout(); plt.savefig(out_path); plt.close()
def plot_learning_curve(curve, out_path, ylabel="Validation RMSE (sec)"):
    if not curve: return
    plt.figure(); plt.plot(range(1, len(curve)+1), curve)
    plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title("Learning curve")
    plt.tight_layout(); plt.savefig(out_path); plt.close()
def plot_qlearning_trajectory(hist_df: pd.DataFrame, out_path: str, title: str):
    if hist_df.empty or "t_pred" not in hist_df.columns: return
    best = np.minimum.accumulate(hist_df["t_pred"].values)
    plt.figure(); plt.plot(range(1, len(best)+1), best)
    plt.xlabel("Step"); plt.ylabel("Best predicted duration (s)")
    plt.title(title); plt.tight_layout(); plt.savefig(out_path); plt.close()
def plot_topk_bar(topk_profile_df: pd.DataFrame, median_val: float, out_path: str, title: str):
    if topk_profile_df.empty: return
    vals = topk_profile_df["t_pred"].values; idx = np.arange(len(vals)); labels = [str(i+1) for i in range(len(vals))]
    plt.figure(); plt.bar(idx, vals); plt.axhline(median_val)
    plt.xticks(idx, labels); plt.xlabel("Top-k candidates (ranked)")
    plt.ylabel("Predicted duration (s)"); plt.title(title + " (line = dataset median)")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

class DiscreteQLearner:
    def __init__(self, value_spaces: Dict[str, List[Any]], predict_fn, gamma=0.9, alpha=0.2, eps=0.2, seed=42):
        self.value_spaces = value_spaces; self.predict_fn = predict_fn
        self.gamma = gamma; self.alpha = alpha; self.eps = eps
        self.rng = random.Random(seed); self.Q = {}
        self.param_order = list(value_spaces.keys())

    def _state_from_indices(self, idxs: Tuple[int,...]) -> Dict[str, Any]:
        return {p: self.value_spaces[p][i] for p, i in zip(self.param_order, idxs)}
    def _initial_state(self) -> Tuple[int,...]:
        return tuple((len(self.value_spaces[p])//2 if len(self.value_spaces[p])>1 else 0) for p in self.param_order)
    def _neighbors(self, state: Tuple[int,...]):
        acts = []
        for j,p in enumerate(self.param_order):
            n = len(self.value_spaces[p])
            if n<=1: continue
            if state[j]-1>=0: acts.append((p,-1))
            if state[j]+1<n: acts.append((p,+1))
        return acts
    def _apply(self, state: Tuple[int,...], action: Tuple[str,int]) -> Tuple[int,...]:
        p,d = action; j = self.param_order.index(p); n = len(self.value_spaces[p])
        new_idx = max(0, min(n-1, state[j]+d))
        s2 = list(state); s2[j]=new_idx; return tuple(s2)
    def _qrow(self, s: Tuple[int,...]):
        if s not in self.Q: self.Q[s] = {a:0.0 for a in self._neighbors(s)}
        return self.Q[s]
    def greedy_action(self, s): 
        row = self._qrow(s); 
        return None if not row else max(row.items(), key=lambda kv: kv[1])[0]
    def eps_greedy_action(self, s):
        row = self._qrow(s)
        if not row: return None
        return self.rng.choice(list(row.keys())) if self.rng.random()<self.eps else self.greedy_action(s)
    def run(self, base_template: pd.DataFrame, max_episodes=50, max_steps=20, min_rel_improve=0.10):
        s_best = self._initial_state(); conf_best = self._state_from_indices(s_best)
        t_best = float(self.predict_fn(self._inject_conf(base_template, conf_best))[0])
        history = []
        for ep in range(max_episodes):
            s = s_best; t_prev = t_best; improved=False
            for step in range(max_steps):
                a = self.eps_greedy_action(s)
                if a is None: break
                s2 = self._apply(s,a); conf2 = self._state_from_indices(s2)
                t_new = float(self.predict_fn(self._inject_conf(base_template, conf2))[0])
                reward = (t_prev - t_new) / max(t_prev, 1e-9)
                row = self._qrow(s); q_sa = row[a]
                row2 = self._qrow(s2); best_next_q = max(row2.values()) if row2 else 0.0
                row[a] = q_sa + self.alpha * (reward + self.gamma * best_next_q - q_sa)
                s = s2; t_prev = t_new
                rel_impr = (t_best - t_prev)/max(t_best,1e-9) if t_best>0 else 0.0
                history.append({"episode":ep,"step":step,"t_pred":t_prev, **conf2, "rel_improve_from_best":rel_impr})
                if t_prev < t_best - 1e-9: t_best = t_prev; s_best = s2; improved=True
                if rel_impr < min_rel_improve and step>0: break
            if not improved: self.eps = min(0.4, self.eps + 0.02)
        import pandas as pd
        hist_df = pd.DataFrame(history)
        return hist_df, {"best_pred_s": t_best, "best_conf": self._state_from_indices(s_best)}
    def _inject_conf(self, tpl: pd.DataFrame, conf: Dict[str, Any]) -> pd.DataFrame:
        row = tpl.copy()
        for k,v in conf.items(): row[k]=v
        return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="out_v3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--qepisodes", type=int, default=50)
    ap.add_argument("--qsteps", type=int, default=20)
    ap.add_argument("--model", choices=["torch","mlp","rf"], default=None, help="Which model to use")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True); plots_dir = os.path.join(args.out,"plots"); os.makedirs(plots_dir, exist_ok=True)
    random.seed(args.seed); np = __import__("numpy"); np.random.seed(args.seed)

    df_raw = load_dataset(args.csv)
    X, y_sec = make_feature_frame(df_raw)
    y_log = np.log1p(y_sec.values)

    num_cols = ["executor_cores","executor_instances","driver_cores","memory_fraction",
                "executor_memory_mb","driver_memory_mb","broadcast_block_mb","max_in_flight_mb",
                "shuffle_file_buffer_mb","input_bytes"]
    cat_cols = ["profile","io_codec"]
    bin_cols = ["shuffle_compress_b","spill_compress_b"]

    X_tr, X_te, y_tr_sec, y_te_sec, y_tr_log, y_te_log = train_test_split(X, y_sec, y_log, test_size=0.2, random_state=args.seed, stratify=X["profile"])
    X_tr, X_va, y_tr_sec, y_va_sec, y_tr_log, y_va_log = train_test_split(X_tr, y_tr_sec, y_tr_log, test_size=0.2, random_state=args.seed, stratify=X_tr["profile"])

    pre = build_preprocessor(X_tr, num_cols, cat_cols, bin_cols)
    Xtr = pre.transform(X_tr); Xva = pre.transform(X_va); Xte = pre.transform(X_te)

    # Baseline: per-profile median
    med_by_prof = X_tr.assign(duration_s=y_tr_sec).groupby("profile")["duration_s"].median().to_dict()
    base_pred = X_te["profile"].map(med_by_prof).values
    base_metrics = {
        "baseline_MAE": float(mean_absolute_error(y_te_sec, base_pred)),
        "baseline_RMSE": float(rmse(y_te_sec.values, base_pred)),
        "baseline_R2": float(r2_score(y_te_sec, base_pred))
    }

    mode = args.model or ("torch" if TORCH_AVAILABLE else "mlp")
    if mode == "torch":
        model, val_curve = train_torch(Xtr, y_tr_log, Xva, y_va_sec, epochs=args.epochs, seed=args.seed, verbose=True)
    elif mode == "mlp":
        model, val_curve = train_sklearn_mlp(Xtr, y_tr_log, Xva, y_va_sec)
    else:
        model, val_curve = train_rf(Xtr, y_tr_log, Xva, y_va_sec)

    def predict_seconds_np(Xp):
        if mode == "torch":
            with torch.no_grad():
                y_log_pred = model(torch.from_numpy(Xp.astype(np.float32))).cpu().numpy().reshape(-1)
        else:
            y_log_pred = model.predict(Xp).reshape(-1)
        y_sec = np.expm1(y_log_pred); y_sec = np.clip(y_sec, 0.0, None)
        return y_sec

    y_hat_te = predict_seconds_np(Xte)
    metrics = {
        "MAE": float(mean_absolute_error(y_te_sec, y_hat_te)),
        "RMSE": float(rmse(y_te_sec.values, y_hat_te)),
        "R2": float(r2_score(y_te_sec, y_hat_te)),
        "MAPE": float(mape(y_te_sec.values, y_hat_te)),
        "n_train": int(len(X_tr)), "n_val": int(len(X_va)), "n_test": int(len(X_te)),
        "mode": mode, **base_metrics
    }
    print("[Metrics]", json.dumps(metrics, indent=2))

    plot_pred_vs_actual(y_te_sec.values, y_hat_te, os.path.join(plots_dir, "pred_vs_actual_test.png"))
    plot_residuals(y_te_sec.values, y_hat_te, os.path.join(plots_dir, "residuals_test.png"))
    if val_curve: plot_learning_curve(val_curve, os.path.join(plots_dir, "learning_curve.png"))

    tm = TrainedModel(pre=pre, model=model, mode=mode,
                      feature_cols=list(X.columns), num_cols=num_cols, cat_cols=cat_cols, bin_cols=bin_cols)
    tm.save(args.out)

    spaces = {
        "executor_instances": sorted(X["executor_instances"].unique().tolist()),
        "executor_memory_mb": sorted(X["executor_memory_mb"].unique().tolist()),
        "driver_memory_mb": sorted(X["driver_memory_mb"].unique().tolist()),
        "memory_fraction": sorted(X["memory_fraction"].unique().tolist()),
        "shuffle_compress_b": sorted(X["shuffle_compress_b"].unique().tolist()),
        "spill_compress_b": sorted(X["spill_compress_b"].unique().tolist()),
        "broadcast_block_mb": sorted(X["broadcast_block_mb"].unique().tolist()),
        "max_in_flight_mb": sorted(X["max_in_flight_mb"].unique().tolist()),
        "shuffle_file_buffer_mb": sorted(X["shuffle_file_buffer_mb"].unique().tolist()),
        "io_codec": sorted(X["io_codec"].unique().tolist()),
    }
    base_common = {
        "executor_cores": int(X["executor_cores"].median()) if "executor_cores" in X else 2,
        "executor_instances": int(X["executor_instances"].median()) if "executor_instances" in X else 2,
        "driver_cores": int(X["driver_cores"].median()) if "driver_cores" in X else 1,
        "memory_fraction": float(X["memory_fraction"].median()) if "memory_fraction" in X else 0.6,
        "executor_memory_mb": float(X["executor_memory_mb"].median()),
        "driver_memory_mb": float(X["driver_memory_mb"].median()),
        "broadcast_block_mb": float(X["broadcast_block_mb"].median()),
        "max_in_flight_mb": float(X["max_in_flight_mb"].median()),
        "shuffle_file_buffer_mb": float(X["shuffle_file_buffer_mb"].median()),
        "shuffle_compress_b": int(X["shuffle_compress_b"].mode()[0]),
        "spill_compress_b": int(X["spill_compress_b"].mode()[0]),
        "io_codec": str(X["io_codec"].mode()[0]),
        "input_bytes": float(X["input_bytes"].median() if "input_bytes" in X else 0)
    }

    suggestions_all = []; report = {"metrics": metrics, "qlearning": {}}
    ref = df_raw.groupby("profile")["duration_s"].median().to_dict()

    for profile in sorted(X["profile"].unique().tolist()):
        base_row = pd.DataFrame([{**base_common, "profile": profile}], columns=X.columns)
        q = DiscreteQLearner(spaces, predict_fn=lambda df: tm.predict_seconds(df), eps=0.2, seed=args.seed)
        hist_df, best = q.run(base_row, max_episodes=args.qepisodes, max_steps=args.qsteps, min_rel_improve=0.10)
        hist_df["profile"] = profile
        hist_path = os.path.join(args.out, f"qlearning_history_{profile}.csv"); hist_df.to_csv(hist_path, index=False)
        plot_qlearning_trajectory(hist_df, os.path.join(plots_dir, f"q_traj_{profile}.png"), title=f"Q-learning best trajectory ({profile})")
        suggestions_all.append(hist_df); report["qlearning"][profile] = best

    sug_df = pd.concat(suggestions_all, ignore_index=True) if suggestions_all else pd.DataFrame()
    topk = pd.DataFrame()
    if not sug_df.empty:
        topk = (sug_df.sort_values(["profile","t_pred"]).groupby("profile").head(10)).copy()
        keep_cols = ["profile","t_pred"] + list(spaces.keys())
        topk = topk[keep_cols].drop_duplicates()
        topk["speedup_vs_median_%"] = topk.apply(
            lambda r: (ref[r["profile"]] - r["t_pred"]) / ref[r["profile"]] * 100.0 if ref[r["profile"]]>0 else 0.0, axis=1
        )
    topk_path = os.path.join(args.out, "qlearning_suggestions.csv"); topk.to_csv(topk_path, index=False)
    if not topk.empty:
        for prof, g in topk.groupby("profile"):
            plot_topk_bar(g.sort_values("t_pred"), ref.get(prof, 0.0), os.path.join(plots_dir, f"q_top10_{prof}.png"),
                          title=f"Top-10 predicted configs ({prof})")

    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Saved artifacts to: {args.out}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
