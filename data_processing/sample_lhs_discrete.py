#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在 data_processing/ 目录下批量读取 .csv：
- 最后一列视为性能，其余为配置列
- 统计每个配置维度的所有可能取值（由 CSV 中出现过的取值确定）
- 基于离散值集合执行拉丁超立方采样（LHS）
- 生成 5%（small）和 10%（medium）两份子集，并从原数据中拼上性能
- 输出到同目录：<stem>_small.csv 与 <stem>_medium.csv
"""

import os
import math
import glob
import random
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def get_config_and_perf_columns(df: pd.DataFrame) -> Tuple[List[str], str]:
    """前 n-1 列为配置，最后一列为性能。"""
    if df.shape[1] < 2:
        raise ValueError("CSV 至少需要包含 2 列（配置 + 性能）。")
    cols = list(df.columns)
    perf_col = cols[-1]
    config_cols = cols[:-1]
    return config_cols, perf_col


def discover_value_space(df: pd.DataFrame, config_cols: List[str]) -> Dict[str, List]:
    """
    提取每个配置维度的所有可能取值（按出现过的唯一值集合）。
    为了可复现，做稳定排序：数字按数值升序，其它按字符串升序。
    """
    value_space = {}
    for c in config_cols:
        vals = pd.unique(df[c])
        # 稳定排序：尝试数值排序，失败则按字符串
        try:
            vals = np.array(sorted(vals.astype(float), key=float))
            # 保持原类型
            if pd.api.types.is_integer_dtype(df[c]):
                vals = vals.astype(int)
        except Exception:
            vals = np.array(sorted(map(str, vals)))
        value_space[c] = list(vals)
    return value_space


def lhs_discrete_indices(n_samples: int, levels: int, rng: np.random.Generator) -> np.ndarray:
    """
    生成单个维度的“离散 LHS”索引（范围 [0, levels-1]）。
    思路：
    - 先在 [0,1) 上做标准 LHS（n_samples 个分层区间，随机抖动）
    - 再将连续值映射到离散等级：floor(u * levels)
    - 为避免同一等级过度聚集，打乱并做一次轻微均衡
    """
    # 标准 LHS（连续）
    cut = np.linspace(0, 1, n_samples + 1)
    u = rng.random(n_samples)  # 抖动
    pts = cut[:-1] + u * (cut[1:] - cut[:-1])  # 每个分层里取一点
    rng.shuffle(pts)  # LHS 置换

    # 映射到离散等级
    idx = np.floor(pts * levels).astype(int)
    idx = np.clip(idx, 0, levels - 1)

    # 简单再均衡：尽量覆盖更多等级
    # 如果 levels >= n_samples，尽可能使用不同等级
    if levels >= n_samples:
        # 取一个不重复的排列（长度 n_samples）
        perm = rng.choice(levels, size=n_samples, replace=False)
        # 和 idx 混合：前半用 perm，后半保留 idx
        half = n_samples // 2
        idx[:half] = perm[:half]
        rng.shuffle(idx)
    else:
        # 如果离散等级少于样本数，保证每个等级至少出现一次
        counts = {k: 0 for k in range(levels)}
        for i in idx:
            counts[i] += 1
        missing = [k for k, v in counts.items() if v == 0]
        if missing:
            # 用出现次数最多的等级替换一部分为缺失等级
            overfull = sorted(counts.items(), key=lambda x: -x[1])
            over_idxs = [k for k, v in overfull for _ in range(v - 1) if v > 1]
            rng.shuffle(over_idxs)
            for m, rep in zip(missing, over_idxs):
                # 找到一个可替换位置
                pos = int(np.where(idx == rep)[0][0])
                idx[pos] = m
    return idx


def lhs_discrete(value_space: Dict[str, List], n_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    在“离散的配置值集合”上执行 LHS。
    返回：包含各配置列的 DataFrame（不含性能）。
    """
    rng = np.random.default_rng(seed)
    cols = list(value_space.keys())
    arrays = []
    for c in cols:
        levels = len(value_space[c])
        if levels == 0:
            raise ValueError(f"维度 {c} 的可能取值集合为空。")
        idx = lhs_discrete_indices(n_samples, levels, rng)
        vals = [value_space[c][i] for i in idx]
        arrays.append(vals)
    data = {c: arrays[i] for i, c in enumerate(cols)}
    return pd.DataFrame(data)


def attach_performance(sample_cfgs: pd.DataFrame, df_full: pd.DataFrame,
                       config_cols: List[str], perf_col: str,
                       seed: int = 42) -> pd.DataFrame:
    """
    将采样得到的配置与原表做“精确匹配”以拼上性能。
    若出现匹配不到的配置（说明该组合在原表中不存在），
    则从原表中随机抽样补齐到相同规模（保证结果行数不变）。
    """
    # 精确匹配（内连接）
    merged = sample_cfgs.merge(df_full[config_cols + [perf_col]], on=config_cols, how='inner')
    # 如有重复配置（采样可能重复），去重
    merged = merged.drop_duplicates(subset=config_cols, keep='first')

    # 如果匹配数量不足，回退：从原表中补齐
    needed = sample_cfgs.shape[0] - merged.shape[0]
    if needed > 0:
        # 从原表中取未被选中的配置行
        used_keys = set(tuple(row) for row in merged[config_cols].to_numpy())
        pool = df_full[~df_full[config_cols].apply(tuple, axis=1).isin(used_keys)]
        if pool.empty:
            pool = df_full  # 兜底：允许重复
        rng = np.random.default_rng(seed)
        extra = pool.sample(n=min(needed, len(pool)), random_state=seed)
        extra = extra[config_cols + [perf_col]]
        merged = pd.concat([merged, extra], ignore_index=True)

        # 若仍不足（极端情况下），再从全表放回采样补齐
        still_need = sample_cfgs.shape[0] - merged.shape[0]
        if still_need > 0:
            extra2 = df_full.sample(n=still_need, replace=True, random_state=seed)[config_cols + [perf_col]]
            merged = pd.concat([merged, extra2], ignore_index=True)

    return merged.reset_index(drop=True)


def process_single_csv(csv_path: str, small_ratio: float = 0.05, medium_ratio: float = 0.25, large_ratio: float = 0.50,
                       seed: int = 42) -> None:
    print(f"\n处理文件: {csv_path}")
    df = pd.read_csv(csv_path)
    config_cols, perf_col = get_config_and_perf_columns(df)
    print(f"配置列: {config_cols}")
    print(f"性能列: {perf_col}")

    # 发现每个维度的所有可能取值
    value_space = discover_value_space(df, config_cols)
    print("每个维度取值规模：")
    for k, v in value_space.items():
        print(f"  - {k}: {len(v)} 个取值")

    n = df.shape[0]
    n_small = max(1, math.floor(n * small_ratio))
    n_medium = max(1, math.floor(n * medium_ratio))
    n_large = max(1, math.floor(n * large_ratio))
    print(f"原始行数: {n} -> small: {n_small}, medium: {n_medium}, large: {n_large}")

    # LHS 采样（只采样配置），再与原表匹配以拼上性能
    small_cfgs = lhs_discrete(value_space, n_small, seed=seed)
    small_df = attach_performance(small_cfgs, df, config_cols, perf_col, seed=seed)

    medium_cfgs = lhs_discrete(value_space, n_medium, seed=seed + 1)
    medium_df = attach_performance(medium_cfgs, df, config_cols, perf_col, seed=seed + 1)

    large_cfgs = lhs_discrete(value_space, n_large, seed=seed + 2)
    large_df = attach_performance(large_cfgs, df, config_cols, perf_col, seed=seed + 2)

    # 输出
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.dirname(csv_path)
    out_small = os.path.join(out_dir, f"{stem}_small.csv")
    out_medium = os.path.join(out_dir, f"{stem}_medium.csv")
    out_large = os.path.join(out_dir, f"{stem}_large.csv")

    small_df.to_csv(out_small, index=False)
    medium_df.to_csv(out_medium, index=False)
    large_df.to_csv(out_large, index=False)

    print(f"已保存：{out_small}")
    print(f"已保存：{out_medium}")
    print(f"已保存：{out_large}")


def main():
    parser = argparse.ArgumentParser(description="对 data_processing 下 CSV 执行离散 LHS 采样，生成 small(5%) & medium(10%) 子集。")
    parser.add_argument("--dir", type=str, default="../data_processing",
                        help="包含 CSV 的目录（默认：data_processing）")
    parser.add_argument("--small", type=float, default=0.05, help="small 比例（默认 0.05）")
    parser.add_argument("--medium", type=float, default=0.25, help="medium 比例（默认 0.10）")
    parser.add_argument("--large", type=float, default=0.50, help="medium 比例（默认 0.10）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"未在目录 {args.dir} 中找到 .csv 文件。")

    for f in csv_files:
        process_single_csv(f, small_ratio=args.small, medium_ratio=args.medium, large_ratio=args.large, seed=args.seed)


if __name__ == "__main__":
    main()
