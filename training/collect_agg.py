#!/usr/bin/env python3
import argparse, glob, pandas as pd, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="./out/**/wordcount_agg.csv", help="glob pattern to search CSVs")
    ap.add_argument("--out", default="./out/dataset_wordcount_large.csv")
    ap.add_argument("--drop-dupes", action="store_true", help="drop duplicate configs")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern, recursive=True))
    if not files:
        print("No files found by pattern:", args.pattern)
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = os.path.abspath(f)
            dfs.append(df)
        except Exception as e:
            print("WARN: failed to read", f, e)

    df = pd.concat(dfs, ignore_index=True)
    if args.drop_dupes:
        keys = ["executors","executor_cores","executor_memory_gb","driver_cores","driver_memory_gb",
                "parallelism","shuffle_partitions","rdd_compress","io_compression_codec","input_gb","scale"]
        df = df.sort_values(["median_sec"]).drop_duplicates(subset=keys, keep="first")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print("OK:", args.out, "rows:", len(df), "from files:", len(files))

if __name__ == "__main__":
    main()
