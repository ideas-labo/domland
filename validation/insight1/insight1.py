import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict


RESULTS_DIR = Path("results")
SAVE_DIR = Path("insight1_visualization")
SUMMARY_DIR = SAVE_DIR / "aggregation"
BUDGET = 1000
SELECTED_ALGORITHMS = ["HC", "HC_option"]

DISPLAY_NAMES = {
    "HC":        "Vanilla HC",
    "HC_option": "Priority HC",
}


def process_regret(F: np.ndarray, budget: int, y_opt: float, system: str) -> np.ndarray:
    """Calculate regret curve and pad/truncate it to `budget` length."""
    best_so_far = (
        np.maximum.accumulate(F.ravel()) if system == "---" else np.minimum.accumulate(F.ravel())
    )
    raw_regret = (y_opt - best_so_far) if system == "---" else (best_so_far - y_opt)

    if raw_regret.size < budget:
        padded = np.full(budget, raw_regret[-1], dtype=float)
        padded[: raw_regret.size] = raw_regret
        return padded
    return raw_regret[:budget]


optima_path = RESULTS_DIR / "optima.json"
if not optima_path.is_file():
    raise FileNotFoundError(f"Cannot find {optima_path}")
with optima_path.open("r") as fh:
    optima = json.load(fh)  # type: Dict[str, Dict[str, float]]


all_convergence_data = {}  # type: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]]

for algorithm in SELECTED_ALGORITHMS:
    algo_path = RESULTS_DIR / algorithm
    if not algo_path.is_dir():
        print(f"[WARN] Algorithm folder not found: {algo_path}")
        continue

    all_convergence_data[algorithm] = {}

    for system, workloads in optima.items():
        system_path = algo_path / system
        if not system_path.is_dir():
            continue

        all_convergence_data[algorithm][system] = {}

        for workload, y_opt in workloads.items():
            if algorithm == "GA":
                prefix = f"{algorithm}_pop20_{system}_{workload}_i"
            else:
                prefix = f"{algorithm}_{system}_{workload}_i"

            npz_files = sorted(p for p in system_path.glob(f"{prefix}*.npz"))
            if not npz_files:
                print(f"[WARN] No results for {algorithm}/{system}/{workload}")
                continue

            regrets = []
            for fp in npz_files:
                data = np.load(fp)
                F = data["F"]
                regrets.append(process_regret(F, BUDGET, y_opt, system))

            if regrets:
                arr = np.stack(regrets)
                all_convergence_data[algorithm][system][workload] = {
                    "mean": arr.mean(axis=0),
                    "std": arr.std(axis=0),
                }

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 20,
    "axes.titlesize": 18,
    "legend.fontsize": 18,
    "lines.linewidth": 2,
})

cmap = plt.get_cmap("tab10")
COLORS = [cmap(i) for i in range(len(SELECTED_ALGORITHMS))]
LINESTYLES = ["-", "-"]
MARKERS = ["o", "*"]

# ----------------------------------------------------------------------------
#  Part 1 – per‑workload plots (original behaviour)
# ----------------------------------------------------------------------------
print("[INFO] Generating per‑workload plots …")

for system, workloads in optima.items():
    for workload in workloads.keys():
        system_dir = SAVE_DIR / system
        system_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 6))
        collected = []

        for algorithm in SELECTED_ALGORITHMS:
            if workload not in all_convergence_data.get(algorithm, {}).get(system, {}):
                continue
            collected.append(all_convergence_data[algorithm][system][workload]["mean"][:BUDGET])

        if not collected:
            continue
        flat = np.concatenate(collected)
        reg_min, reg_max = flat.min(), flat.max()

        for idx, algorithm in enumerate(SELECTED_ALGORITHMS):
            stats = all_convergence_data.get(algorithm, {}).get(system, {}).get(workload)
            if stats is None:
                continue
            mean_reg, std_reg = stats["mean"], stats["std"]
            x = np.arange(BUDGET)
            if reg_max > reg_min:
                mean_norm = (mean_reg - reg_min) / (reg_max - reg_min)
                std_norm = std_reg / (reg_max - reg_min)
            else:
                mean_norm = np.zeros_like(mean_reg)
                std_norm = np.zeros_like(std_reg)

            plt.plot(
                x,
                mean_norm,
                label=DISPLAY_NAMES.get(algorithm, algorithm),
                color=COLORS[idx],
                linestyle=LINESTYLES[idx],
                marker=MARKERS[idx],
                markersize=10,
                markevery=10
            )

            plt.fill_between(x, mean_norm - std_norm, mean_norm + std_norm, color=COLORS[idx], alpha=0.15)



        plt.xlabel("No. of Evaluations", fontsize=28)
        plt.ylabel("Normalized Regret", fontsize=28)
        # plt.title(f"{system} – {workload}", fontsize=20)
        plt.xlim(0, 103)
        # if system.upper() == "X264":
        #     plt.ylim(0.6, 1.05)
        # elif system.upper() == "H2":
        #     plt.ylim(0.3, 1.05)
        # elif system.upper() == "JUMP3R":
        #     plt.ylim(0.2, 1.05)
        # elif system.upper() == "KANZI":
        #     plt.ylim(0.6, 1.05)
        # elif system.upper() == "XZ":
        #     plt.ylim(0.2, 1.05)

        plt.ylim(-0.1, 1.1)
        plt.legend(loc="best", frameon=False, ncol=1, fontsize=22)

        out_path = system_dir / f"{workload}_normalized.pdf"
        plt.savefig(out_path, bbox_inches="tight", format="pdf")
        plt.close()
        print(f"  [OK] {out_path}")

# ----------------------------------------------------------------------------
#  Part 2 – system‑level aggregation plots (re‑normalise per‑workload first) ★
# ----------------------------------------------------------------------------
print("[INFO] Generating system‑level aggregation plots …")
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

for system in optima.keys():

    per_algo_normalised = {algo: [] for algo in SELECTED_ALGORITHMS}

    for workload in optima[system].keys():
        # calculate min/max of all algorithms for this workload
        workload_means = []
        for algo in SELECTED_ALGORITHMS:
            stats = all_convergence_data.get(algo, {}).get(system, {}).get(workload)
            if stats is None:
                continue
            workload_means.append(stats["mean"][:BUDGET])

        if not workload_means:
            continue
        w_min, w_max = np.min(workload_means), np.max(workload_means)
        scale = w_max - w_min
        if scale == 0:
            scale = 1.0

        for algo in SELECTED_ALGORITHMS:
            stats = all_convergence_data.get(algo, {}).get(system, {}).get(workload)
            if stats is None:
                continue
            mean_norm = (stats["mean"][:BUDGET] - w_min) / scale
            per_algo_normalised[algo].append(mean_norm)

    aggregated_stats = {}
    for algo, curves in per_algo_normalised.items():
        if not curves:
            continue
        stack = np.vstack(curves)  # shape: (n_workloads, budget)
        aggregated_stats[algo] = {
            "mean": stack.mean(axis=0),
            "std": stack.std(axis=0),
        }

    if not aggregated_stats:
        print(f"  [SKIP] No data for system {system}")
        continue


    x = np.arange(BUDGET)
    plt.figure(figsize=(8, 6))

    for idx, algo in enumerate(SELECTED_ALGORITHMS):
        stats = aggregated_stats.get(algo)
        if stats is None:
            continue
        mean_reg, std_reg = stats["mean"], stats["std"]
        plt.plot(
            x,
            mean_reg,
            label=DISPLAY_NAMES.get(algo, algo),
            color=COLORS[idx],
            linestyle=LINESTYLES[idx],
            linewidth=2,
            marker=MARKERS[idx],
            markersize=15,
            markevery=10
        )
        plt.fill_between(x, mean_reg - std_reg, mean_reg + std_reg, color=COLORS[idx], alpha=0.15)

    plt.xlabel("No. of Evaluations", fontsize=28)
    plt.ylabel("Normalized Regret", fontsize=28)
    plt.title(system.upper(), fontsize=28)
    plt.xlim(0, 103)
    if system.upper() == "X264":
        plt.ylim(0.6, 1.05)
    elif system.upper() == "H2":
        plt.ylim(0.3, 1.05)
    elif system.upper() == "JUMP3R":
        plt.ylim(0.2, 1.05)
    elif system.upper() == "KANZI":
        plt.ylim(0.7, 1.05)
    elif system.upper() == "XZ":
        plt.ylim(0.2, 1.05)

    # plt.ylim(0, 1)
    plt.legend(loc="best", frameon=False, ncol=1, fontsize=22)

    out_path = SUMMARY_DIR / f"{system}.pdf"
    plt.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"  [OK] {out_path}")

print("[DONE] All figures written to", SAVE_DIR)
