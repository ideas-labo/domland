import os
import math
import glob
import random
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def get_config_and_perf_columns(df: pd.DataFrame) -> Tuple[List[str], str]:
    """ first n-1 columns are configurations, last column is performance."""
    if df.shape[1] < 2:
        raise ValueError("CSV should have at least two columns (config + performance).")
    cols = list(df.columns)
    perf_col = cols[-1]
    config_cols = cols[:-1]
    return config_cols, perf_col


def discover_value_space(df: pd.DataFrame, config_cols: List[str]) -> Dict[str, List]:

    value_space = {}
    for c in config_cols:
        vals = pd.unique(df[c])
        try:
            vals = np.array(sorted(vals.astype(float), key=float))
            if pd.api.types.is_integer_dtype(df[c]):
                vals = vals.astype(int)
        except Exception:
            vals = np.array(sorted(map(str, vals)))
        value_space[c] = list(vals)
    return value_space


def lhs_discrete_indices(n_samples: int, levels: int, rng: np.random.Generator) -> np.ndarray:

    cut = np.linspace(0, 1, n_samples + 1)
    u = rng.random(n_samples)
    pts = cut[:-1] + u * (cut[1:] - cut[:-1])
    rng.shuffle(pts)

    idx = np.floor(pts * levels).astype(int)
    idx = np.clip(idx, 0, levels - 1)

    if levels >= n_samples:
        perm = rng.choice(levels, size=n_samples, replace=False)
        half = n_samples // 2
        idx[:half] = perm[:half]
        rng.shuffle(idx)
    else:
        counts = {k: 0 for k in range(levels)}
        for i in idx:
            counts[i] += 1
        missing = [k for k, v in counts.items() if v == 0]
        if missing:
            overfull = sorted(counts.items(), key=lambda x: -x[1])
            over_idxs = [k for k, v in overfull for _ in range(v - 1) if v > 1]
            rng.shuffle(over_idxs)
            for m, rep in zip(missing, over_idxs):
                pos = int(np.where(idx == rep)[0][0])
                idx[pos] = m
    return idx


def lhs_discrete(value_space: Dict[str, List], n_samples: int, seed: int = 42) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    cols = list(value_space.keys())
    arrays = []
    for c in cols:
        levels = len(value_space[c])
        if levels == 0:
            raise ValueError(f"column {c} has no available values for sampling.")
        idx = lhs_discrete_indices(n_samples, levels, rng)
        vals = [value_space[c][i] for i in idx]
        arrays.append(vals)
    data = {c: arrays[i] for i, c in enumerate(cols)}
    return pd.DataFrame(data)


def attach_performance(sample_cfgs: pd.DataFrame, df_full: pd.DataFrame,
                       config_cols: List[str], perf_col: str,
                       seed: int = 42) -> pd.DataFrame:

    merged = sample_cfgs.merge(df_full[config_cols + [perf_col]], on=config_cols, how='inner')
    merged = merged.drop_duplicates(subset=config_cols, keep='first')

    needed = sample_cfgs.shape[0] - merged.shape[0]
    if needed > 0:
        used_keys = set(tuple(row) for row in merged[config_cols].to_numpy())
        pool = df_full[~df_full[config_cols].apply(tuple, axis=1).isin(used_keys)]
        if pool.empty:
            pool = df_full
        rng = np.random.default_rng(seed)
        extra = pool.sample(n=min(needed, len(pool)), random_state=seed)
        extra = extra[config_cols + [perf_col]]
        merged = pd.concat([merged, extra], ignore_index=True)

        still_need = sample_cfgs.shape[0] - merged.shape[0]
        if still_need > 0:
            extra2 = df_full.sample(n=still_need, replace=True, random_state=seed)[config_cols + [perf_col]]
            merged = pd.concat([merged, extra2], ignore_index=True)

    return merged.reset_index(drop=True)


def process_single_csv(csv_path: str, small_ratio: float = 0.05, medium_ratio: float = 0.25, large_ratio: float = 0.50,
                       seed: int = 42) -> None:
    print(f"\nhandled file: {csv_path}")
    df = pd.read_csv(csv_path)
    config_cols, perf_col = get_config_and_perf_columns(df)
    print(f"config columns: {config_cols}")
    print(f"perf columns: {perf_col}")

    value_space = discover_value_space(df, config_cols)
    print("scale of each column：")
    for k, v in value_space.items():
        print(f"  - {k}: {len(v)} values")

    n = df.shape[0]
    n_small = max(1, math.floor(n * small_ratio))
    n_medium = max(1, math.floor(n * medium_ratio))
    n_large = max(1, math.floor(n * large_ratio))
    print(f"original rows: {n} -> small: {n_small}, medium: {n_medium}, large: {n_large}")

    small_cfgs = lhs_discrete(value_space, n_small, seed=seed)
    small_df = attach_performance(small_cfgs, df, config_cols, perf_col, seed=seed)

    medium_cfgs = lhs_discrete(value_space, n_medium, seed=seed + 1)
    medium_df = attach_performance(medium_cfgs, df, config_cols, perf_col, seed=seed + 1)

    large_cfgs = lhs_discrete(value_space, n_large, seed=seed + 2)
    large_df = attach_performance(large_cfgs, df, config_cols, perf_col, seed=seed + 2)

    stem = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.dirname(csv_path)
    out_small = os.path.join(out_dir, f"{stem}_small.csv")
    out_medium = os.path.join(out_dir, f"{stem}_medium.csv")
    out_large = os.path.join(out_dir, f"{stem}_large.csv")

    small_df.to_csv(out_small, index=False)
    medium_df.to_csv(out_medium, index=False)
    large_df.to_csv(out_large, index=False)

    print(f"saved：{out_small}")
    print(f"saved：{out_medium}")
    print(f"saved：{out_large}")


def main():
    parser = argparse.ArgumentParser(description=" handle CSV files in data_processing with discrete LHS sampling to create small (5%), medium (10%), and large (50%) subsets.")
    parser.add_argument("--dir", type=str, default="../data_processing",
                        help="directory containing CSV files (default: data_processing)")
    parser.add_argument("--small", type=float, default=0.05, help="small（default 0.05）")
    parser.add_argument("--medium", type=float, default=0.25, help="medium（default 0.25）")
    parser.add_argument("--large", type=float, default=0.50, help="large （default 0.50）")
    parser.add_argument("--seed", type=int, default=42, help="random seed（default 42）")
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"can't find .csv file in {args.dir} directory.")

    for f in csv_files:
        process_single_csv(f, small_ratio=args.small, medium_ratio=args.medium, large_ratio=args.large, seed=args.seed)


if __name__ == "__main__":
    main()
