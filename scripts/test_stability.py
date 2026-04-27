"""
Проверка стабильности рекомендаций: делает N запросов к /recommend
и показывает, насколько стабильны предсказанное время и все 16 параметров.

Использование:
  python scripts/test_stability.py                        # 10 запросов, дефолтные параметры
  python scripts/test_stability.py --n 20 --profile small
  python scripts/test_stability.py --workers 2 --cores 4 --ram 8
"""

import argparse
import json
import statistics
import urllib.request
from collections import defaultdict

BASE_URL = "http://localhost:8001"

# Все 16 параметров конфигурации, разбитые по группам
PARAM_GROUPS = {
    "Ресурсы executors/driver": [
        "executor_cores", "executor_memory_mb", "executor_instances",
        "driver_cores", "driver_memory_mb",
    ],
    "Память": [
        "memory_fraction", "memory_storageFraction",
    ],
    "Сжатие": [
        "shuffle_compress", "spill_compress", "broadcast_compress", "rdd_compress", "io_codec",
    ],
    "Буферы I/O": [
        "shuffle_file_buffer_kb", "broadcast_block_mb", "maxSizeInFlight_mb", "rpc_message_maxSize",
    ],
}
ALL_PARAMS = [p for params in PARAM_GROUPS.values() for p in params]


def recommend(body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/recommend",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


def predict_default(workers, cores, ram, profile) -> float:
    body = {
        "job_type": "pagerank", "profile": profile,
        "topology_workers": workers, "topology_worker_cores": cores,
        "topology_worker_mem_gb": ram,
        "executor_cores": 1, "executor_memory_mb": 1024, "executor_instances": 2,
        "driver_cores": 1, "driver_memory_mb": 1024,
        "memory_fraction": 0.6, "memory_storageFraction": 0.5,
        "shuffle_compress": 1, "spill_compress": 1,
        "shuffle_file_buffer_kb": 32, "broadcast_block_mb": 4,
        "broadcast_compress": 1, "maxSizeInFlight_mb": 48,
        "rpc_message_maxSize": 128, "rdd_compress": 0, "io_codec": "lz4",
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/predict",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)["predicted_runtime_s"]


def fmt_val(v) -> str:
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _collect_results(body: dict, n: int) -> list:
    results = []
    print(f"{'№':>3}  {'Время,с':>8}  {'Ускор.':>7}  {'cores':>5}  {'mem,МБ':>7}  {'inst':>4}  {'codec':>6}  {'shuf_compr':>10}")
    print("-" * 70)
    for i in range(n):
        top1 = recommend(body)["recommendations"][0]
        results.append(top1)
        cfg = top1["config"]
        shuf = "Да" if cfg["shuffle_compress"] else "Нет"
        print(f"{i+1:3d}  {top1['predicted_runtime_s']:8.1f}  "
              f"×{top1['predicted_speedup_vs_default']:6.3f}  "
              f"{cfg['executor_cores']:5d}  {cfg['executor_memory_mb']:7d}  "
              f"{cfg['executor_instances']:4d}  {cfg['io_codec']:>6}  {shuf:>10}")
    return results


def _print_runtime_stats(runtimes: list, speedups: list) -> float:
    print()
    print("=" * 70)
    print("ИТОГ: стабильность топ-1 рекомендации")
    print("=" * 70)
    if len(runtimes) > 1:
        cv = statistics.stdev(runtimes) / statistics.mean(runtimes) * 100
        print(f"  Время (с):   min={min(runtimes):.1f}  max={max(runtimes):.1f}  "
              f"среднее={statistics.mean(runtimes):.1f}  σ={statistics.stdev(runtimes):.1f}  CV={cv:.1f}%")
    else:
        cv = 0.0
        print(f"  Время (с):   {runtimes[0]:.1f}")
    print(f"  Ускорение:   min=×{min(speedups):.3f}  max=×{max(speedups):.3f}  "
          f"среднее=×{statistics.mean(speedups):.3f}")
    return cv


def _print_param_distribution(results: list) -> None:
    param_counts: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for r in results:
        for p in ALL_PARAMS:
            param_counts[p][r["config"][p]] += 1

    n = len(results)
    print()
    print("Распределение всех 16 параметров конфигурации:")
    for group_name, params in PARAM_GROUPS.items():
        print(f"\n  [{group_name}]")
        for p in params:
            vals = sorted(param_counts[p].items(), key=lambda x: -x[1])
            _, dominant_cnt = vals[0]
            dominant_pct = dominant_cnt / n * 100
            parts = "  ".join(f"{fmt_val(v)}→{c}/{n} ({c/n*100:.0f}%)" for v, c in vals)
            if dominant_pct >= 80:
                marker = " ✓"
            elif dominant_pct >= 50:
                marker = "  "
            else:
                marker = " ?"
            print(f"    {p:<28} {parts}{marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Количество запросов")
    parser.add_argument("--profile", default="large", choices=["small", "large"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cores", type=int, default=6)
    parser.add_argument("--ram", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=1, dest="top_k")
    parser.add_argument("--model", default="rf", choices=["rf", "dnn", "ql"],
                        help="Модель для рекомендации: rf | dnn | ql (default: rf)")
    args = parser.parse_args()

    body = {
        "job_type": "pagerank",
        "input": {"profile": args.profile},
        "topology": {
            "workers": args.workers,
            "worker_cores": args.cores,
            "worker_memory_gb": args.ram,
        },
        "preferences": {"return_top_k": args.top_k},
        "model_name": args.model,
    }

    print(f"Топология: {args.workers} воркеров × {args.cores} ядер × {args.ram} ГБ  |  профиль={args.profile}  |  модель={args.model}  |  n={args.n}")
    print()
    default_t = predict_default(args.workers, args.cores, args.ram, args.profile)
    print(f"Дефолтный конфиг (cores=1, mem=1024 МБ, instances=2): {default_t:.1f} с")
    print()

    results = _collect_results(body, args.n)
    runtimes = [r["predicted_runtime_s"] for r in results]
    speedups = [r["predicted_speedup_vs_default"] for r in results]
    cv = _print_runtime_stats(runtimes, speedups)
    _print_param_distribution(results)

    print()
    if cv < 5:
        print(f"ВЫВОД: рекомендации стабильны (CV={cv:.1f}% < 5%)")
    else:
        print(f"ВЫВОД: высокая вариативность (CV={cv:.1f}%), модель чувствительна к стохастике кандидатов")


if __name__ == "__main__":
    main()
