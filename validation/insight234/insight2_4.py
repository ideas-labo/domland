import os
import json
import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR = "results"
optima_path = os.path.join(RESULTS_DIR, "optima.json")
with open(optima_path, "r") as f:
    optima = json.load(f)  # {system: {workload: best_value}}


def process_regret(F, budget, _yopt, system):

    best_so_far = np.maximum.accumulate(F.ravel()) if system == "---" else np.minimum.accumulate(F.ravel())
    _regret = (_yopt - best_so_far) if system == "---" else (best_so_far - _yopt)
    current_length = len(_regret)
    if current_length < budget:
        last_value = _regret[-1]
        padded_regret = np.full(budget, last_value)
        padded_regret[:current_length] = _regret
        return padded_regret

    return _regret[:budget]


all_convergence_data = {}
budget = 1000


for algorithm in os.listdir(RESULTS_DIR):
    algo_path = os.path.join(RESULTS_DIR, algorithm)
    if not os.path.isdir(algo_path) or algorithm == "optima.json":
        continue

    all_convergence_data[algorithm] = {}

    for system, workloads in optima.items():
        system_path = os.path.join(algo_path, system)
        if not os.path.isdir(system_path):
            continue

        all_convergence_data[algorithm][system] = {}

        for workload, _yopt in workloads.items():
            if algorithm == "GA":
                workload_files = [
                    f for f in os.listdir(system_path) if f.startswith(f"{algorithm}_pop20_{system}_{workload}_i")
                ]
            else:
                workload_files = [
                    f for f in os.listdir(system_path) if f.startswith(f"{algorithm}_{system}_{workload}_i")
                ]
            workload_files.sort()

            if not workload_files:
                print(f"Warning: No results found for {algorithm}/{system}/{workload}")
                continue

            regret_list = []

            for filename in workload_files:
                file_path = os.path.join(system_path, filename)
                _data = np.load(file_path)
                F = _data["F"]

                _regret = process_regret(F, budget, _yopt, system)
                regret_list.append(_regret)

            if regret_list:
                regret_array = np.array(regret_list)
                mean_regret = np.mean(regret_array, axis=0)
                std_regret = np.std(regret_array, axis=0)

                all_convergence_data[algorithm][system][workload] = {
                    "mean": mean_regret,
                    "std": std_regret
                }


plt.rcParams.update({
    # "text.usetex": True,
    # "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 20,
    "axes.titlesize": 18,
    # "xtick.labelsize": 18,
    # "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "lines.linewidth": 2,
})


cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(7)]
linestyles = ["-", "-", "-", "-", "-", "-", "-"]
markers = ["o", "*", "^", "s", ".", ".", "."]

SAVE_DIR = "insight2_4_visualization"

# selected_algorithms = ["HC", "GA", "IRACE", "RS", "SA", "FLASH"]
selected_algorithms = ['HC', 'GA', 'TPE', 'RS']


for system, workloads in optima.items():
    for wl_idx, workload in enumerate(workloads.keys(), start=1):

        system_dir = os.path.join(SAVE_DIR, system)
        os.makedirs(system_dir, exist_ok=True)

        plt.figure(figsize=(8, 6))
        all_regrets = []
        for idx, (algorithm, color, linestyle, marker) in enumerate(
                zip(selected_algorithms, colors, linestyles, markers)):
            if workload not in all_convergence_data[algorithm].get(system, {}):
                continue

            mean_regret = all_convergence_data[algorithm][system][workload]["mean"]
            all_regrets.append(mean_regret[0:1000])

        if all_regrets:
            all_regrets = np.concatenate(all_regrets)
            regret_min = np.min(all_regrets)
            regret_max = np.max(all_regrets)


        for idx, (algorithm, color, linestyle, marker) in enumerate(
                zip(selected_algorithms, colors, linestyles, markers)):
            # if algorithm != "HC" and algorithm != "RS":
            #     continue
            if workload not in all_convergence_data[algorithm].get(system, {}):
                continue

            mean_regret = all_convergence_data[algorithm][system][workload]["mean"]
            std_regret = all_convergence_data[algorithm][system][workload]["std"]
            x = np.arange(len(mean_regret))

            if regret_max > regret_min:
                regret_norm = (mean_regret[0:1000] - regret_min) / (regret_max - regret_min)
                lower_norm = ((mean_regret[:1000] - std_regret[:1000]) - regret_min) / (regret_max - regret_min)
                upper_norm = ((mean_regret[:1000] + std_regret[:1000]) - regret_min) / (regret_max - regret_min)

                lower_norm = np.clip(lower_norm, 0.0, 1.0)
                upper_norm = np.clip(upper_norm, 0.0, 1.0)
            else:
                regret_norm = np.full_like(mean_regret[0:1000], 0.5)
                lower_norm = upper_norm = regret_norm


            plt.plot(x[0:1000], regret_norm, label=algorithm, color=color, linestyle=linestyle,
                     marker=marker, markersize=10, markevery=10)
            plt.fill_between(x[:1000],
                             lower_norm,
                             upper_norm,
                             color=color,
                             alpha=0.15)


        plt.xlabel("No. of Evaluations", fontsize=28)
        plt.ylabel("Normalized Regret", fontsize=28)
        plt.legend(loc="upper right", frameon=False, ncol=1, fontsize=22)

        plt.xlim(0, 103)
        system_name = system.upper()
        plt.title(f'{system_name} (W{wl_idx})', fontsize=28)
        # plt.ylim(0, 0.5)
        # plt.show()
        save_path = os.path.join(system_dir, f"{workload}_normalized.pdf")
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.close()
        print(f"  [OK] {save_path}")

# ----------------------------------------------------------------------------
#  Part 2 – system-level aggregation plots  ★
# ----------------------------------------------------------------------------
# print("[INFO] Generating system-level summary plots …")
#
# summary_dir = os.path.join(SAVE_DIR, "summary")
# os.makedirs(summary_dir, exist_ok=True)
#
# for system in optima.keys():
#
#     per_algo_norm = {algo: [] for algo in selected_algorithms}
#     for workload in optima[system]:
#
#         workload_means = [
#             all_convergence_data[algo][system][workload]["mean"][:budget]
#             for algo in selected_algorithms
#             if workload in all_convergence_data.get(algo, {}).get(system, {})
#         ]
#         if not workload_means:
#             continue
#         w_min, w_max = np.min(workload_means), np.max(workload_means)
#         scale = w_max - w_min if w_max > w_min else 1.0
#
#         for algo in selected_algorithms:
#             stats = all_convergence_data.get(algo, {}).get(system, {}).get(workload)
#             if stats is None:
#                 continue
#             mean_norm = (stats["mean"][:budget] - w_min) / scale
#             per_algo_norm[algo].append(mean_norm)
#
#     aggregated = {}
#     for algo, curves in per_algo_norm.items():
#         if curves:
#             stack = np.vstack(curves)
#             aggregated[algo] = {"mean": stack.mean(axis=0), "std": stack.std(axis=0)}
#
#     if not aggregated:
#         print(f"  [SKIP] {system} (no data)")
#         continue
#
#     x = np.arange(budget)
#     plt.figure(figsize=(8, 6))
#
#     for idx, (algo, color, ls, mk) in enumerate(zip(selected_algorithms, colors, linestyles, markers)):
#         if algo not in aggregated:
#             continue
#         mean = aggregated[algo]["mean"]
#         std = aggregated[algo]["std"]
#         plt.plot(
#             x,
#             mean,
#             label=algo,
#             color=color,
#             linestyle=ls,
#             marker=mk,
#             markersize=10,
#             markevery=10,
#         )
#         plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
#
#     plt.xlabel("No. of Evaluations", fontsize=28)
#     plt.ylabel("Normalized Regret", fontsize=28)
#     plt.title(f"{system.upper()} (Aggregated)", fontsize=28)
#     plt.xlim(0, 103)
#     plt.legend(loc="upper right", frameon=False, ncol=1, fontsize=22)
#
#     out_path = os.path.join(summary_dir, f"{system}_summary.pdf")
#     plt.savefig(out_path, bbox_inches="tight", format="pdf")
#     plt.close()
#     print(f"  [OK] {out_path}")
