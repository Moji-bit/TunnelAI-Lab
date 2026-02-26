# dataset/merge_csv.py
import glob
import pandas as pd
import os


def merge_all_csv(
    raw_dir="data/raw",
    out_path="data/raw/all_runs.csv",
):
    files = sorted(glob.glob(os.path.join(raw_dir, "stau_case_*.csv")))
    dfs = [pd.read_csv(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(out_path, index=False)
    print("✅ Merged to:", out_path)
    return out_path


if __name__ == "__main__":
    merge_all_csv()