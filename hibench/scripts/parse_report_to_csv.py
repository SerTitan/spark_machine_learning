#!/usr/bin/env python3
import re, statistics, csv, sys
from pathlib import Path

runs_file = Path("/tmp/wc_runs.txt")
out_csv   = Path("/tmp/wordcount_runs.csv")

if not runs_file.exists():
    print(f"ERR: {runs_file} not found. Run wordcount_9runs.sh first.", file=sys.stderr)
    sys.exit(1)

rows, durations = [], []

with runs_file.open() as f:
    for line in f:
        parts = re.split(r"\s+", line.strip())
        if len(parts) < 6:
            continue
        bench, engine, scale = parts[0], parts[1], parts[2]
        try:
            duration = float(parts[5])  # seconds
        except Exception:
            continue
        rows.append([bench, engine, scale, duration])
        durations.append(duration)

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["benchmark","engine","scale","duration_sec"])
    w.writerows(rows)

if durations:
    durations.sort()
    median = statistics.median(durations)
    print(f"OK: parsed {len(durations)} runs. median={median:.3f}s")
    print(f"CSV: {out_csv}")
else:
    print("WARN: no durations parsed")
