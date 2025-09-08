#!/usr/bin/env python3
import csv, re, sys, json, statistics
from pathlib import Path

runs_file = Path("/tmp/wc_runs.txt")
report    = Path("/opt/hibench/report/hibench.report")
meta_json = Path("/tmp/wc_meta.json")

out_runs  = Path("/tmp/wordcount_runs.csv")
out_agg   = Path("/tmp/wordcount_agg.csv")

rows = []   # raw runs
meta = {}

def load_meta():
    if meta_json.exists():
        try:
            with meta_json.open() as f:
                return json.load(f)
        except Exception as e:
            print(f"WARN: failed to parse {meta_json}: {e}", file=sys.stderr)
    return {}

def parse_wc_runs_line(line: str):
    # "WordCount,Spark,tiny,12.345"
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) == 4 and parts[0].lower().startswith("wordcount"):
        bench, engine, scale, dur = parts
        try:
            return bench, engine, scale, float(dur)
        except:
            return None
    return None

def parse_report_line(line: str):
    # "WordCount spark tiny 12.345 ..."
    parts = [p.strip() for p in re.split(r"[\t ]+", line.strip())]
    if len(parts) >= 4 and parts[0].lower().startswith("wordcount"):
        bench, engine, scale, dur = parts[0], parts[1], parts[2], parts[3]
        try:
            return bench, engine, scale, float(dur)
        except:
            m = re.search(r"([0-9]+(\.[0-9]+)?)", dur)
            if m:
                return bench, engine, scale, float(m.group(1))
    return None

def emit_runs_csv(rows, meta):
    fields = [
        "benchmark","engine","scale","run_idx","duration_sec",
        "executors","executor_cores","executor_memory_gb",
        "driver_cores","driver_memory_gb","parallelism","shuffle_partitions",
        "rdd_compress","io_compression_codec","input_gb","is_default","spark_conf"
    ]
    out_runs.parent.mkdir(parents=True, exist_ok=True)
    with out_runs.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for idx, (bench, engine, scale, dur) in enumerate(rows, start=1):
            rec = {
                "benchmark": bench,
                "engine":    engine,
                "scale":     scale,
                "run_idx":   idx,
                "duration_sec": f"{dur:.3f}",
                "executors": meta.get("executors"),
                "executor_cores": meta.get("executor_cores"),
                "executor_memory_gb": meta.get("executor_memory_gb"),
                "driver_cores": meta.get("driver_cores"),
                "driver_memory_gb": meta.get("driver_memory_gb"),
                "parallelism": meta.get("parallelism"),
                "shuffle_partitions": meta.get("shuffle_partitions"),
                "rdd_compress": meta.get("rdd_compress"),
                "io_compression_codec": meta.get("io_compression_codec"),
                "input_gb": meta.get("input_gb"),
                "is_default": meta.get("is_default", 0),
                "spark_conf": meta.get("spark_conf"),
            }
            w.writerow(rec)

def emit_agg_csv(rows, meta):
    durations = [dur for _,_,_,dur in rows]
    if not durations:
        print("WARN: no durations for aggregation.", file=sys.stderr)
        return
    durations_sorted = sorted(durations)
    def pct(p):
        k = (len(durations_sorted)-1) * (p/100.0)
        f = int(k); c = min(f+1, len(durations_sorted)-1)
        if f == c: return durations_sorted[f]
        return durations_sorted[f] + (durations_sorted[c]-durations_sorted[f])*(k-f)
    agg = {
        "benchmark": rows[0][0],
        "engine": rows[0][1],
        "scale": rows[0][2],
        "n": len(durations),
        "mean_sec": f"{statistics.mean(durations):.3f}",
        "median_sec": f"{statistics.median(durations):.3f}",
        "p50_sec": f"{pct(50):.3f}",
        "p90_sec": f"{pct(90):.3f}",
        "p99_sec": f"{pct(99):.3f}",
        "std_sec": f"{(statistics.pstdev(durations) if len(durations)>1 else 0.0):.3f}",
        "executors": meta.get("executors"),
        "executor_cores": meta.get("executor_cores"),
        "executor_memory_gb": meta.get("executor_memory_gb"),
        "driver_cores": meta.get("driver_cores"),
        "driver_memory_gb": meta.get("driver_memory_gb"),
        "parallelism": meta.get("parallelism"),
        "shuffle_partitions": meta.get("shuffle_partitions"),
        "rdd_compress": meta.get("rdd_compress"),
        "io_compression_codec": meta.get("io_compression_codec"),
        "input_gb": meta.get("input_gb"),
        "is_default": meta.get("is_default", 0),
    }
    fields = list(agg.keys())
    with out_agg.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(agg)

# ===== main =====
meta = load_meta()

# 1) Prefer wc_runs.txt
if runs_file.exists():
    with runs_file.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = parse_wc_runs_line(line)
            if rec:
                rows.append(rec)

# 2) Fallback: hibench.report
if not rows and report.exists():
    with report.open() as f:
        for line in f:
            if not line.strip() or line.lower().startswith("benchmark"):
                continue
            rec = parse_report_line(line)
            if rec:
                rows.append(rec)

if not rows:
    print("WARN: no runs parsed (neither /tmp/wc_runs.txt nor hibench.report).", file=sys.stderr)

emit_runs_csv(rows, meta)
emit_agg_csv(rows, meta)

print(f"OK: parsed {len(rows)} run(s).")
print(f"CSV (runs): {out_runs}")
print(f"CSV (agg) : {out_agg}")
