import os
import pandas as pd
from scipy import stats
import numpy as np



def calculate_a12(group1, group2):
    n1, n2 = len(group1), len(group2)
    ranked = stats.rankdata(np.concatenate((group1, group2)))
    rank1 = ranked[:n1]
    a12 = ((rank1.sum() - (n1 * (n1 + 1) / 2)) / (n1 * n2))
    return a12 if a12 > 0.5 else 1 - a12


def collect_data(
    systems,
    algorithms,
    runs,
    base_path='../results',
    dst_folder='insight5'
):
    os.makedirs(dst_folder, exist_ok=True)

    for system in systems:
        results_list = []

        env_path = os.path.join(base_path, system,
                                algorithms[0], 'optimized_pop_perf_run_0')
        environments = sorted(
            f.replace('_perf.csv', '') for f in os.listdir(env_path)
            if f.endswith('_perf.csv')
        )

        for env in environments:
            algo_data = {}
            for algo in algorithms:
                values = []
                for run in range(runs):
                    file_path = os.path.join(
                        base_path, system, algo,
                        f'optimized_pop_perf_run_{run}',
                        f'{env}_perf.csv'
                    )
                    if os.path.exists(file_path):
                        run_df = pd.read_csv(file_path, header=None, nrows=1)
                        values.extend(run_df.iloc[0].values)
                algo_data[algo] = values

            if all(len(algo_data[a]) > 0 for a in algorithms):

                stat, p_value = stats.mannwhitneyu(
                    algo_data[algorithms[0]], algo_data[algorithms[1]]
                )
                a12_value = calculate_a12(
                    algo_data[algorithms[0]], algo_data[algorithms[1]]
                )

                if p_value < 0.05 and a12_value > 0.56:
                    mean0 = np.mean(algo_data[algorithms[0]])
                    mean1 = np.mean(algo_data[algorithms[1]])
                    better_is_lower = system != 'h2'
                    if (mean0 < mean1 and better_is_lower) or (mean0 > mean1 and not better_is_lower):
                        label = 1
                    else:
                        label = -1
                else:
                    label = 0


                stats_dict = {}
                for algo in algorithms:
                    mean = np.mean(algo_data[algo])
                    std = np.std(algo_data[algo], ddof=1)
                    stats_dict[f'{algo}_Mean'] = mean
                    stats_dict[f'{algo}_Std'] = std

                    stats_dict[algo] = f'{mean:.3f} ({std:.3f})'


                results = {
                    'Environment': env,
                    **{algo: stats_dict[algo] for algo in algorithms},
                    **{f'{algo}_Mean': stats_dict[f'{algo}_Mean']
                       for algo in algorithms},
                    **{f'{algo}_Std': stats_dict[f'{algo}_Std']
                       for algo in algorithms},
                    'Wilcoxon_p': p_value,
                    'A12': a12_value,
                    'Label': label
                }
                results_list.append(results)


        out_path = os.path.join(dst_folder,
                                f'{system}_{algorithms[0]}_vs_{algorithms[1]}.csv')
        pd.DataFrame(results_list).to_csv(out_path, index=False)
        print(f'✅  {system}: saved {out_path}')


def main():
    systems = ['batik', 'dconvert', 'h2', 'jump3r',
               'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    compared_algorithms = ['Transfer-GA', 'Vanilla-GA']
    runs = 30
    collect_data(systems, compared_algorithms, runs)


if __name__ == '__main__':
    main()
