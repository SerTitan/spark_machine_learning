#!/usr/bin/env python3
import csv, re, sys
from pathlib import Path

runs_file = Path("/tmp/wc_runs.txt")
report    = Path("/opt/hibench/report/hibench.report")
out_csv   = Path("/tmp/wordcount_runs.csv")

rows = []

def parse_wc_runs_line(line: str):
    # Наш формат: WordCount,Spark,<scale>,<duration_sec>
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) == 4 and parts[0].lower().startswith("wordcount"):
        bench, engine, scale, dur = parts
        try:
            return bench, engine, scale, float(dur)
        except:
            return None
    return None

def parse_report_line(line: str):
    # Псевдо-HiBench: "WordCount\tSpark\t<scale>\t<duration>"
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

# 1) Пробуем wc_runs.txt (наш основной источник)
if runs_file.exists():
    with runs_file.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = parse_wc_runs_line(line)
            if rec:
                rows.append(rec)

# 2) Если вдруг wc_runs.txt пустой — пробуем report
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

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["benchmark","engine","scale","duration_sec"])
    for bench, engine, scale, dur in rows:
        w.writerow([bench, engine, scale, f"{dur:.3f}"])

print(f"OK: parsed {len(rows)} runs.")
print(f"CSV: {out_csv}")
