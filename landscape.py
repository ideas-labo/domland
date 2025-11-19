import os

import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import pairwise_distances
import seaborn as sns
from scipy.sparse import issparse
import pandas as pd
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
import random
from utils import Utils
from sklearn.neighbors import NearestNeighbors
from typing import Optional


class Landscape:
    def __init__(self, system_name):
        self.system_name = system_name
        self.system_path = os.path.join('datasets', system_name)
        self.system_path_2d = os.path.join(self.system_path, 'workload_2d')

        self.utils = Utils(system_name)
        # Load performance and configuration data
        self.perf_dict = self.utils.load_perf_data(self.system_path)
        self.config_dict = self.utils.load_config_data(self.system_path)

        # Define all required directories
        self.paths = {
            "landscape_metrics": ('landscape_results', system_name, 'landscape_metrics'),
            "landscape_metrics_FDC": ('landscape_results', system_name, 'landscape_metrics', 'FDC'),
            "landscape_metrics_BoA": ('landscape_results', system_name, 'landscape_metrics', 'BoA'),
            "landscape_metrics_AC": ('landscape_results', system_name, 'landscape_metrics', 'AC'),
            "system_path_2d": (self.system_path_2d,)
        }

        # Generate full path strings
        self.paths = {key: os.path.join(*value) for key, value in self.paths.items()}

        # Create directories if they do not exist
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)


    '''---------------------Landscape Metrics--------------------------'''
    def calculate_fdc(self):
        """
        Calculate the FDC for targeted system (with multiple workloads) and save the results to {system_name}_FDC.csv.
        """
        fdc_results = []

        for workload, config_data in self.config_dict.items():
            perf_data = self.perf_dict[workload]

            # Particularly, it is a maximization optimization problem for system 'h2'
            if self.system_name == 'h2':
                # Transform to a minimization optimization problem
                perf_data = -perf_data
                global_optimum_value = np.min(perf_data)
            else:
                global_optimum_value = np.min(perf_data)

            # Find all global optimum indices that have the same performance value
            global_optimum_indices = np.where(perf_data == global_optimum_value)[0]

            # Calculate the integer Hamming distance to the closest global optimal configuration
            config_array = config_data.values
            distances = np.min(
                [np.sum(config_array != config_array[opt_idx], axis=1) for opt_idx in global_optimum_indices], axis=0
            )

            # Get the mean values of fitness (performance space) and distances (config space)
            f_mean = np.mean(perf_data)
            d_mean = np.mean(distances)

            # Calculate the covariance C_{FD}
            covariance_fd = np.mean((perf_data - f_mean) * (distances - d_mean))

            # Calculate the std for fitness and distances
            sigma_f = np.std(perf_data)
            sigma_d = np.std(distances)

            # Calculate FDC and avoid dividing by zero if the standard deviation is zero
            if sigma_f == 0 or sigma_d == 0:
                fdc_value = 0
                print("sigma_f = 0 or sigma_d = 0")
            else:
                fdc_value = covariance_fd / (sigma_f * sigma_d)

            fdc_results.append([self.system_name, workload, fdc_value])

        # Save the results to sys_FDC.csv file
        fdc_df = pd.DataFrame(fdc_results, columns=['system', 'workload', 'FDC'])
        output_file = os.path.join(self.paths["landscape_metrics_FDC"], f'{self.system_name}_FDC.csv')
        fdc_df.to_csv(output_file, index=False)
        print(f"The FDC results for {self.system_name} are saved to {output_file}")

    # def calculate_local_optima(self):
    #     """
    #     Identify the local optimal configs for target system (with multiple workloads) based on adaptive hamming
    #     distance.
    #     """
    #     local_optima_dict = {}
    #     hamming_distance_dict = {}
    #
    #     for workload, config_data in self.config_dict.items():
    #         configs = pd.DataFrame(config_data)
    #         performances = self.perf_dict[workload]
    #         config_dim = configs.shape[1]
    #
    #         # make sure the average neighbor of all the samples no smaller than dim/2 (target_neighbors).
    #         target_neighbors = config_dim  # int(config_dim / 2)
    #
    #         # The maximum Hamming distance is dynamically adjusted so that it is not set too small, resulting in many
    #         # individuals having no neighbors within this distance. In that case, increase Hamming Distance step size to
    #         # find Neighbors. Therefore, the first thing is to determine the radius.
    #         max_hamming_distance = 1
    #         while True:
    #             # Calculate Hamming distances (note: *config_dim is for obtaining the number of different bits)
    #             hamming_distances = pairwise_distances(configs, metric='hamming') * config_dim
    #             neighbors_mask = (hamming_distances > 0) & (hamming_distances <= max_hamming_distance)
    #             neighbors_count = np.sum(neighbors_mask, axis=1)  # Calculate the No. of neighbors for each config
    #             avg_neighbors = np.mean(neighbors_count)
    #
    #             # If the average No. of neighbors is greater than or equal to a predetermined value, the loop stops
    #             if avg_neighbors >= target_neighbors:
    #                 break
    #             else:
    #                 # Zoom in on finding neighbors' step lengths
    #                 max_hamming_distance += 1
    #
    #         # Record used hamming distance for identifying neighbors
    #         hamming_distance_dict[workload] = max_hamming_distance
    #         local_optima_with_neighbors = []
    #         isolated_local_optima = []
    #
    #         # Iterate through each configuration and check if it is local optima
    #         for i in range(len(configs)):
    #             neighbor_indices = np.where(neighbors_mask[i])[0]
    #             if len(neighbor_indices) == 0:
    #                 isolated_local_optima.append(i)
    #             else:
    #                 neighbor_performances = performances[neighbor_indices]
    #                 if self.system_name == 'h2':
    #                     if performances[i] >= max(neighbor_performances):
    #                         local_optima_with_neighbors.append(i)
    #                 else:
    #                     if performances[i] <= min(neighbor_performances):
    #                         local_optima_with_neighbors.append(i)
    #
    #         # Merging isolated local optima and local optima with neighbors
    #         all_local_optima_indices = local_optima_with_neighbors + isolated_local_optima
    #         local_optima_dict[workload] = all_local_optima_indices
    #
    #     return local_optima_dict, hamming_distance_dict

    def calculate_local_optima(self):
        """
        Identify local optima for each workload using a memory-friendly
        radius neighbors graph under Hamming distance.
        - 不再构造 NxN 稠密矩阵，改用稀疏邻接图
        - 动态增大汉明半径，直到平均邻居数 >= target_neighbors
        """
        local_optima_dict = {}
        hamming_distance_dict = {}

        for workload, config_data in self.config_dict.items():

            configs_df = pd.DataFrame(config_data).copy()
            performances = np.asarray(self.perf_dict[workload])
            config_dim = configs_df.shape[1]

            # encode categorical/string columns to category codes
            for c in configs_df.columns:
                if not (pd.api.types.is_integer_dtype(configs_df[c]) or
                        pd.api.types.is_bool_dtype(configs_df[c]) or
                        pd.api.types.is_float_dtype(configs_df[c])):
                    configs_df[c] = pd.Categorical(configs_df[c]).codes

            X = configs_df.to_numpy()
            n = X.shape[0]

            # targeted average neighbor count
            target_neighbors = config_dim

            # dynamically adjust max_hamming_distance to ensure average neighbors >= target_neighbors
            max_hamming_distance = 1
            avg_neighbors = 0.0
            # use 'brute' + metric='hamming' to build radius neighbors, return sparse graph (without self-loop)
            nn = NearestNeighbors(metric='hamming', algorithm='brute')
            nn.fit(X)

            while max_hamming_distance <= config_dim:
                radius = max_hamming_distance / config_dim  # hamming distance ratio
                try:
                    # new version of sklearn, support include_self
                    graph = nn.radius_neighbors_graph(
                        X, radius=radius, mode='connectivity', include_self=False
                    )
                except TypeError:
                    # old version of sklearn, not support include_self
                    graph = nn.radius_neighbors_graph(
                        X, radius=radius, mode='connectivity'
                    )
                    if issparse(graph):
                        graph.setdiag(0)
                        graph.eliminate_zeros()

                avg_neighbors = graph.nnz / X.shape[0]
                if avg_neighbors >= target_neighbors:
                    break
                max_hamming_distance += 1

            # if exceeded config_dim, set to config_dim
            hamming_distance_dict[workload] = max_hamming_distance

            # judge local optima based on sparse adjacency
            indptr = graph.indptr
            indices = graph.indices

            local_optima_with_neighbors = []
            isolated_local_optima = []

            minimize = (self.system_name != 'h2')  # h2 is maximization problem

            for i in range(n):
                start, end = indptr[i], indptr[i + 1]
                neigh_idx = indices[start:end]
                if neigh_idx.size == 0:
                    isolated_local_optima.append(i)
                    continue

                neigh_perf = performances[neigh_idx]
                if minimize:
                    if performances[i] <= np.min(neigh_perf):
                        local_optima_with_neighbors.append(i)
                else:
                    if performances[i] >= np.max(neigh_perf):
                        local_optima_with_neighbors.append(i)

            all_local_optima_indices = local_optima_with_neighbors + isolated_local_optima
            local_optima_dict[workload] = all_local_optima_indices

        return local_optima_dict, hamming_distance_dict


    def visualize_local_optima_distribution(self, save_dir='local_optima_distribution'):
        os.makedirs(os.path.join(save_dir, self.system_name), exist_ok=True)

        local_optima_dict, _ = self.calculate_local_optima()
        for workload, optima_indices in local_optima_dict.items():
            configs = pd.DataFrame(self.config_dict[workload])
            performances = self.perf_dict[workload]

            if self.system_name == 'h2':
                global_opt_idx = np.where(performances == max(performances))[0]
            else:
                global_opt_idx = np.where(performances == min(performances))[0]

            global_opts = configs.iloc[global_opt_idx].values
            local_opts = configs.iloc[optima_indices].values
            local_perf = np.array(performances)[optima_indices]

            hamming_dists = []
            for local in local_opts:
                dists = [np.sum(local != global_opt) for global_opt in global_opts]
                hamming_dists.append(min(dists))

            plt.figure()
            plt.scatter(hamming_dists, local_perf, alpha=0.7)
            plt.xlabel("Hamming Distance to Global Optimum")
            plt.ylabel("Performance (Fitness)")
            plt.title(f"Local Optima Distribution - {workload}")
            plt.grid(True)
            plt.tight_layout()
            file_path = os.path.join(save_dir, self.system_name, f"{workload}.pdf")
            plt.savefig(file_path)
            plt.close()

    def calculate_local_optima_quality(self, local_optima_dict):
        """
        Calculate the quality of the local optimal solutions for target system (of each workload).
        The formula ensures that a higher quality value means local optima are closer to the global optima.

        For maximization:
            quality = (mean(local optima) - mean(all solutions)) / (global optima - mean(all solutions))
        For minimization:
            quality = (mean(all solutions) - mean(local optima)) / (mean(all solutions) - global optima)

        :param local_optima_dict: a dict that records the index of local optima {workload: index_of_local_optima}
        :return: a dict that records the quality of local optima {workload: quality_of_local_optima}
        """

        quality_dict = {}
        for workload, local_optima_indices in local_optima_dict.items():
            performances = self.perf_dict[workload]

            all_mean_value = np.mean(performances)
            local_optima_mean_value = np.mean(performances[local_optima_indices])

            if self.system_name == 'h2':
                global_optimum_value = np.max(performances)
                numerator = local_optima_mean_value - all_mean_value
                denominator = global_optimum_value - all_mean_value
            else:
                global_optimum_value = np.min(performances)
                numerator = all_mean_value - local_optima_mean_value
                denominator = all_mean_value - global_optimum_value

            if denominator != 0:
                quality = numerator / denominator
            else:
                quality = float('inf')
                print(f"Warning: denominator is 0 for workload {workload}")

            quality_dict[workload] = quality

        return quality_dict

    # def calculate_basin(self, local_optima_dict, hamming_distance_dict):
    #     """
    #     Calculate the Basin of Attraction for each local optima
    #     :param local_optima_dict: the dict for which saves the index of local optima
    #     :param hamming_distance_dict: Radius for identify neighbors
    #     :return: Basin of Attraction, the size of basin for each local optima
    #     """
    #     basin_dict = {}  # store the size of basin(of attraction) for each local optima
    #     neutral_dict = {}  # store the ratio of neutral points
    #
    #     for workload, config_data in self.config_dict.items():
    #         configs = pd.DataFrame(config_data)
    #         performances = self.perf_dict[workload]
    #         local_optima_indices = local_optima_dict[workload]
    #         max_hamming_distance = hamming_distance_dict[workload]
    #
    #         # initial basin list for each local optima
    #         basin_dict[workload] = {optima_index: [] for optima_index in local_optima_indices}
    #         neutral_list = []
    #
    #         hamming_distances = pairwise_distances(configs, metric='hamming') * configs.shape[1]
    #
    #         # iterating each non-local optima
    #         for i in range(len(configs)):
    #             if i not in local_optima_indices:
    #                 current_index = i
    #
    #                 while True:
    #                     # retrieve the neighbors of current configuration
    #                     neighbors_mask = (hamming_distances[current_index] > 0) & \
    #                                      (hamming_distances[current_index] <= max_hamming_distance)
    #                     neighbor_indices = np.where(neighbors_mask)[0]
    #
    #                     # all the neighbors' perf are the same as current config -> getting trap to neutrality
    #                     if np.all(performances[neighbor_indices] == performances[current_index]):
    #                         neutral_list.append(i)
    #                         break
    #
    #                     # Check whether local optima exists in neighbors
    #                     local_optima_neighbors = [idx for idx in neighbor_indices if idx in local_optima_indices]
    #                     if local_optima_neighbors:
    #                         # if there are multiple local optimal configs, select the best one.
    #                         if self.system_name == 'h2':
    #                             best_optima = max(local_optima_neighbors, key=lambda idx: performances[idx])
    #                         else:
    #                             best_optima = min(local_optima_neighbors, key=lambda idx: performances[idx])
    #
    #                         # record the start point into the basin of the best local optima
    #                         basin_dict[workload][best_optima].append(i)
    #                         break
    #                     else:
    #                         # if there isn't local optima exist, jump to the best neighbor
    #                         best_neighbor = neighbor_indices[
    #                             np.argmax(performances[neighbor_indices])] if self.system_name == 'h2' \
    #                             else neighbor_indices[np.argmin(performances[neighbor_indices])]
    #
    #                         # update current point
    #                         current_index = best_neighbor
    #
    #         # calculate the ratio of neural points.
    #         neutral_dict[workload] = len(neutral_list) / len(configs)
    #
    #     # calculate the size of basin for each local optima
    #     basin_size_dict = {workload: {optima_index: len(basin) for optima_index, basin in basin_info.items()}
    #                        for workload, basin_info in basin_dict.items()}
    #
    #     # Write to JSON file for next retrieving
    #     output_file = os.path.join(self.paths["landscape_metrics_BoA"], f'{self.system_name}_basin_data.json')
    #
    #     with open(output_file, 'w') as f:
    #         json.dump(basin_size_dict, f, indent=4)
    #     print(f"Basin data saved to {output_file}")
    #
    #     return basin_size_dict

    def calculate_basin(self, local_optima_dict, hamming_distance_dict):
        """
        计算每个局部最优的吸引域大小（Basin of Attraction）
        - 基于汉明半径邻域的稀疏邻接图（避免 OOM）
        - 对每个非局部最优点，按邻域性能做逐步“爬/下山”，直到吸附到某个局部最优；若陷入平坦/无邻居/环，则记为中性点
        返回：basin_size_dict（每个局部最优的盆地大小），并写入 JSON
        """
        basin_dict = {}  # {workload: {opt_idx: [起点i, ...]}}
        neutral_dict = {}  # {workload: ratio_of_neutral}

        for workload, config_data in self.config_dict.items():
            # --- 1) 数据准备：与 calculate_local_optima 一致 ---
            configs_df = pd.DataFrame(config_data).copy()
            performances = np.asarray(self.perf_dict[workload])
            dim = configs_df.shape[1]

            # 分类/字符串 -> category codes（保证汉明比较“相等/不等”）
            for c in configs_df.columns:
                if not (pd.api.types.is_integer_dtype(configs_df[c]) or
                        pd.api.types.is_bool_dtype(configs_df[c]) or
                        pd.api.types.is_float_dtype(configs_df[c])):
                    configs_df[c] = pd.Categorical(configs_df[c]).codes

            X = configs_df.to_numpy()
            n = X.shape[0]

            local_optima_indices = set(local_optima_dict[workload])
            max_hamming_distance = int(hamming_distance_dict[workload])
            # 半径是“比例”：d / dim
            radius = max(1, max_hamming_distance) / dim

            # --- 2) 构建半径邻居稀疏图（不包含自环） ---
            nn = NearestNeighbors(metric='hamming', algorithm='brute')
            nn.fit(X)
            try:
                graph = nn.radius_neighbors_graph(
                    X, radius=radius, mode='connectivity', include_self=False
                )
            except TypeError:
                graph = nn.radius_neighbors_graph(
                    X, radius=radius, mode='connectivity'
                )
                if issparse(graph):
                    graph.setdiag(0)
                    graph.eliminate_zeros()

            indptr = graph.indptr
            indices = graph.indices

            # --- 3) 初始化盆地容器 ---
            basin_dict[workload] = {opt_idx: [] for opt_idx in local_optima_indices}
            neutral_list = []

            minimize = (self.system_name != 'h2')

            # --- 4) 对每个“非局部最优”点，沿邻域做爬/下山搜索 ---
            for i in range(n):
                if i in local_optima_indices:
                    continue

                current = i
                visited = set([current])
                max_steps = dim * 4  # 保护上限，防止极端环/平台导致的长链

                while True:
                    start, end = indptr[current], indptr[current + 1]
                    neigh_idx = indices[start:end]

                    # 无邻居：视为中性点（采样稀疏或极端情况）
                    if neigh_idx.size == 0:
                        neutral_list.append(i)
                        break

                    neigh_perf = performances[neigh_idx]
                    curr_perf = performances[current]

                    # 全部邻居性能与当前相同：平坦平台 -> 中性点
                    if np.all(neigh_perf == curr_perf):
                        neutral_list.append(i)
                        break

                    # 邻居中的局部最优
                    neigh_lopt = [idx for idx in neigh_idx if idx in local_optima_indices]
                    if neigh_lopt:
                        # 选择最佳局部最优
                        if minimize:
                            best_opt = min(neigh_lopt, key=lambda idx: performances[idx])
                        else:
                            best_opt = max(neigh_lopt, key=lambda idx: performances[idx])
                        basin_dict[workload][best_opt].append(i)
                        break

                    # 否则朝“更优邻居”移动
                    if minimize:
                        best_neighbor = neigh_idx[np.argmin(neigh_perf)]
                        # 若最优邻居不更优（数值相等且不在局部最优集合），可能出现环/停滞 → 中性
                        if performances[best_neighbor] >= curr_perf:
                            neutral_list.append(i)
                            break
                    else:
                        best_neighbor = neigh_idx[np.argmax(neigh_perf)]
                        if performances[best_neighbor] <= curr_perf:
                            neutral_list.append(i)
                            break

                    # 防环：走到走过的点，判为中性
                    if best_neighbor in visited:
                        neutral_list.append(i)
                        break

                    visited.add(best_neighbor)
                    current = best_neighbor

                    max_steps -= 1
                    if max_steps <= 0:
                        # 极端保护：步数过长也归为中性
                        neutral_list.append(i)
                        break

            # 中性点占比
            neutral_dict[workload] = len(neutral_list) / n

        # --- 5) 统计盆地大小并落盘 ---
        basin_size_dict = {
            workload: {opt_idx: len(starts) for opt_idx, starts in basin_info.items()}
            for workload, basin_info in basin_dict.items()
        }

        output_file = os.path.join(self.paths["landscape_metrics_BoA"], f'{self.system_name}_basin_data.json')
        os.makedirs(self.paths["landscape_metrics_BoA"], exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(basin_size_dict, f, indent=4)
        print(f"Basin data saved to {output_file}")

        return basin_size_dict


    def save_local_structure(self, local_optima_dict, basin_size_dict, quality_dict, hamming_distance_dict):
        """
        Restore the local structure info into local_structure.csv

        :param hamming_distance_dict: key(workload) values(the radius for identifying neighbors )
        :param local_optima_dict: key(workload) values(the index of local optima)
        :param basin_size_dict: key(workload) values(key(index of local optima),values(basin size))
        :param quality_dict: key(workload) value(quality of local optima)
        """
        output_file = os.path.join(self.paths["landscape_metrics"], 'local_structure.csv')

        with open(output_file, mode='w') as file:
            file.write('system,workload,local_optima_ratio,quality,max_basin,min_basin,hamming_r\n')
            for workload, local_optima_indices in local_optima_dict.items():
                total_configs = len(self.config_dict[workload])
                num_local_optima = len(local_optima_indices)

                # Calculate the ratio of local optima under all sampled configs
                local_optima_ratio = num_local_optima / total_configs

                # Get the quality of local optima for given workload
                quality = quality_dict.get(workload, 0)  # If isn't exist, default 0

                # Calculate the max and min size of basin
                basin_info = basin_size_dict[workload]
                if basin_info:
                    max_basin = max(basin_info.values())
                    min_basin = min(basin_info.values())
                else:
                    max_basin = 0
                    min_basin = 0

                hamming_r = hamming_distance_dict[workload]
                file.write(f'{self.system_name},{workload},{local_optima_ratio},{quality},{max_basin},{min_basin},{hamming_r}\n')

        print(f"Local structure data saved to {output_file}")

    def calculate_global_basin(self, basin_size_dict):
        """
        Calculate the global optimum's basin size (ratio) for each workload and store the results in a CSV file.
        If there are multiple global optima, their basin sizes are summed.
        :param basin_size_dict: Dictionary containing the basin sizes for each workload.
        """
        global_basin_ratios = []

        # Iterate over each workload and calculate the global optima basin size ratio
        for workload, perf_data in self.perf_dict.items():
            total_configs = len(perf_data)

            # Identify the global optima (in case of multiple optima)
            if self.system_name == 'h2':
                global_optimum_indices = np.where(perf_data == np.max(perf_data))[0]  # Maximization for 'h2'
            else:
                global_optimum_indices = np.where(perf_data == np.min(perf_data))[0]  # Minimization for other systems

            # Get the basin size for each global optimum and sum them
            basin_info = basin_size_dict[workload]
            total_basin_size = sum([basin_info.get(f'{idx}', 0) for idx in global_optimum_indices])

            # Get the performance of all global optima
            global_optimum_perfs = perf_data[global_optimum_indices[0]]

            # Calculate the ratio of the global optimum basin size to the total number of configurations
            basin_ratio = total_basin_size / total_configs

            # Store the result: system name, workload, number of global optima,
            # global optima performance, total basin size, and the basin ratio
            global_basin_ratios.append([
                self.system_name, workload, len(global_optimum_indices),
                global_optimum_perfs, total_basin_size, basin_ratio
            ])

        # Create a DataFrame to store the results
        basin_df = pd.DataFrame(global_basin_ratios, columns=[
            'system', 'workload', 'num_global_optima', 'global_optimum_perfs',
            'total_basin_size', 'global_basin_ratio'
        ])

        # Define output file path and store results to a CSV file
        output_file = os.path.join(self.paths["landscape_metrics_BoA"], f'{self.system_name}_global_basin_ratios.csv')
        basin_df.to_csv(output_file, index=False)
        print(f"Global optima basin ratios saved to {output_file}")

    def load_basin_data(self):
        """
        Load the info of basin_size_dict from JSON filee
        :return: basin_size_dict。
        """
        input_file = os.path.join(self.paths["landscape_metrics_BoA"], f'{self.system_name}_basin_data.json')
        with open(input_file, 'r') as f:
            basin_size_dict = json.load(f)

        print(f"Basin data loaded from {input_file}")

        return basin_size_dict

    # @staticmethod
    # def compute_hamming_distances(config_data):
    #     """
    #     Computes the Hamming distances between all configurations.
    #     :param config_data:
    #     :return:
    #     """
    #     configs = pd.DataFrame(config_data)
    #     config_dim = configs.shape[1]
    #     hamming_distances = pairwise_distances(configs, metric='hamming') * config_dim
    #     return hamming_distances


    @staticmethod
    def compute_hamming_distances(config_data,
                                  out_path: Optional[str] = None,
                                  block_rows: int = 4096,
                                  dtype=np.uint16):
        """
        计算所有配置之间的“汉明计数”距离（等价于 pairwise_distances(..., metric='hamming') * D）。
        - 对非数值/分类列做 category 编码，保证“相等/不等”的语义；
        - 用分块计算避免内存爆炸；
        - 如果提供 out_path，则返回一个 NxN 的 memmap（磁盘映射数组）；否则返回常规 NumPy 数组。

        参数
        ----
        config_data : DataFrame 或 可转为 DataFrame 的对象
        out_path    : 如果给出路径，将结果写为 memmap（推荐大数据使用）；否则返回内存数组
        block_rows  : 行分块大小，按你的内存调小/调大（越大越快、越占内存）
        dtype       : 距离矩阵存储类型。汉明计数 ∈ [0, D]，通常用 uint16 足够；D>65535 时改用 uint32。

        返回
        ----
        dist : np.ndarray (N, N) 或 np.memmap (N, N)，元素为“汉明计数”（整数）
        """
        # --- 1) 编码：把非整型/布尔/浮点列编码成类别整数（配置一般是离散的） ---
        df = pd.DataFrame(config_data).copy()
        for c in df.columns:
            if not (pd.api.types.is_integer_dtype(df[c]) or
                    pd.api.types.is_bool_dtype(df[c]) or
                    pd.api.types.is_float_dtype(df[c])):
                df[c] = pd.Categorical(df[c]).codes
        X = df.to_numpy()
        N, D = X.shape

        # 自动选择 dtype（防止维度太大溢出）
        if dtype is None:
            dtype = np.uint16 if D <= np.iinfo(np.uint16).max else np.uint32

        # --- 2) 准备输出矩阵（内存或 memmap） ---
        if out_path:
            dist = np.memmap(out_path, mode='w+', dtype=dtype, shape=(N, N))
        else:
            dist = np.empty((N, N), dtype=dtype)

        # 对角为 0（自身距离为 0）
        np.fill_diagonal(dist, 0)

        # --- 3) 分块计算上三角（并镜像到下三角） ---
        # 公式：Ham(A_i, B_j) = sum_k [X[i,k] != X[j,k]]
        # 用分块避免一次性构造 (R, C, D) 过大张量
        for r0 in range(0, N, block_rows):
            r1 = min(r0 + block_rows, N)
            A = X[r0:r1]  # (R, D)

            # 只算上三角（c0 从 r0 开始）
            for c0 in range(r0, N, block_rows):
                c1 = min(c0 + block_rows, N)
                B = X[c0:c1]  # (C, D)

                # 广播比较：A[:, None, :] vs B[None, :, :]
                # (R, 1, D) != (1, C, D) -> (R, C, D) -> sum over D -> (R, C)
                # 这一块在内存受控（~ R*C*D*1byte），由 block_rows 控制
                neq = (A[:, None, :] != B[None, :, :])
                block = neq.sum(axis=2)

                # 写入上三角
                dist[r0:r1, c0:c1] = block.astype(dtype, copy=False)
                # 镜像到下三角（避免再算一次）
                if r0 != c0:
                    dist[c0:c1, r0:r1] = block.T.astype(dtype, copy=False)

        # memmap 需要 flush；普通 ndarray 忽略即可
        if out_path:
            dist.flush()
        return dist

    @staticmethod
    def traverse_landscape(hamming_distances, config_data):
        """
        Traverses the landscape by visiting all configurations and finding the sequence based on Hamming distance.
        The key is to avoid repeating the same path (A -> B or B -> A).
        """
        num_configs = len(hamming_distances)
        visited_paths = set()  # Store visited paths as a set of tuples (A, B) and (B, A)
        visited_configs = set()  # Track visited configurations
        sequence = []  # To store the sequence of the path

        # Randomly select the starting configuration
        current_index = random.randint(0, num_configs - 1)
        visited_configs.add(current_index)
        sequence.append(current_index)

        while len(visited_configs) < num_configs:
            current_hamming = hamming_distances[current_index]

            # Find neighbors with Hamming distance = 1
            neighbors_mask = (current_hamming > 0) & (current_hamming == 1)
            neighbor_indices = np.where(neighbors_mask)[0]

            # Remove neighbors that have already been traversed in either direction
            neighbor_indices = [idx for idx in neighbor_indices if (current_index, idx) not in visited_paths]

            if not neighbor_indices:
                # Expand the Hamming distance if no neighbors are available within distance 1
                config_dim = config_data.shape[1]  # Get the dimensionality of the configuration
                for dist in range(2, config_dim + 1):  # Expand Hamming distance up to the config's dimensionality
                    neighbors_mask = (current_hamming > 0) & (current_hamming == dist)
                    neighbor_indices = np.where(neighbors_mask)[0]
                    neighbor_indices = [idx for idx in neighbor_indices if (current_index, idx) not in visited_paths]

                    if neighbor_indices:
                        break

            # If multiple unvisited neighbors, pick a random one
            if neighbor_indices:
                next_index = random.choice(neighbor_indices)
                visited_paths.add((current_index, next_index))
                visited_paths.add((next_index, current_index))  # Mark both directions visited
                current_index = next_index
                sequence.append(current_index)
                visited_configs.add(current_index)
            else:
                # This condition shouldn't happen as we ensure to expand the distance.
                raise Exception("No available paths. There may be an error in neighbor identification.")

        return sequence

    @staticmethod
    def autocorrelation_analysis(sequence, performances):
        """
        Calculates the autocorrelation along the sequence generated by the traversal of the landscape.
        """
        # Convert the sequence of indices into performance values
        performance_sequence = [performances[idx] for idx in sequence]

        # Compute the autocorrelation using lag-1 autocorrelation
        performance_mean = np.mean(performance_sequence)
        numerator = 0.0
        denominator = 0.0

        # Calculate autocorrelation starting from index 0 (i, i+1)
        for i in range(len(performance_sequence) - 1):  # Adjust the loop to go up to len-1
            numerator += (performance_sequence[i] - performance_mean) * (performance_sequence[i + 1] - performance_mean)
            denominator += (performance_sequence[i] - performance_mean) ** 2

        if denominator == 0:
            return 0  # Handle edge cases where variance is zero

        autocorrelation = numerator / denominator

        return autocorrelation

    def calculate_autocorrelation(self):
        """
        Runs the full process for all workloads in the system:
        - Compute Hamming distances
        - Traverse the landscape
        - Compute autocorrelation for each workload
        """
        autocorrelation_results = []

        for workload, config_data in self.config_dict.items():
            performances = self.perf_dict[workload]

            # Compute Hamming distances for this workload
            hamming_distances = self.compute_hamming_distances(config_data)

            # Traverse the landscape, generating the sequence of configurations
            sequence = self.traverse_landscape(hamming_distances, config_data)

            # Calculate autocorrelation based on the traversal sequence
            autocorrelation_value = self.autocorrelation_analysis(sequence, performances)

            # Store the result for this workload
            autocorrelation_results.append([self.system_name, workload, autocorrelation_value])

        # Save results to a CSV file
        autocorrelation_df = pd.DataFrame(autocorrelation_results, columns=['system', 'workload', 'autocorrelation'])
        output_file = os.path.join(self.paths["landscape_metrics_AC"], f'{self.system_name}_autocorrelation.csv')
        autocorrelation_df.to_csv(output_file, index=False)
        print(f"Autocorrelation results saved to {output_file}")




    '''----------individual option impact on auto-correlation------------'''

    def calculate_ind_option_impact_on_autocorrelation(self):
        """
        Calculate the FDC sensitivity for each workload based on symmetric subsets of data
        split by each dimension's values while ensuring subsets are symmetric.
        Store the results in a CSV file with each dimension as a separate column.
        """
        sensitivity_results = []

        for workload, config_data in self.config_dict.items():
            perf_data = self.perf_dict[workload]
            row_result = {'system': self.system_name, 'workload': workload}  # Initialize a row for the current workload

            for col in config_data.columns:  # Iterate over each dimension
                unique_values = config_data[col].unique()  # Get unique values of the dimension

                # Divide data into subsets based on the target dimension's values
                subsets = {value: config_data[config_data[col] == value].copy() for value in unique_values}

                # Attach original indices to each subset for future reference
                for value, subset in subsets.items():
                    subset['original_index'] = subset.index

                # Find the smallest subset for iteration
                smallest_subset_value = min(subsets, key=lambda v: len(subsets[v]))
                smallest_subset = subsets[smallest_subset_value]

                # Prepare symmetric subsets
                symmetric_subsets = {value: [] for value in unique_values}

                for _, row in smallest_subset.iterrows():
                    symmetric_rows = []
                    # option's value and corresponding to subset
                    for value, subset in subsets.items():
                        if value != smallest_subset_value:
                            # Match rows from other subsets where all other dimensions are identical
                            matching_rows = subset[
                                (subset.drop(columns=[col, 'original_index']) == row.drop(
                                    labels=[col, 'original_index'])).all(axis=1)
                            ]

                            # If a match is found, add the matching row to the symmetric subset
                            if not matching_rows.empty:
                                symmetric_rows.append(matching_rows.iloc[0])

                    # If symmetric rows are found for all values, add them to the symmetric subsets
                    if len(symmetric_rows) == len(unique_values) - 1:
                        symmetric_subsets[smallest_subset_value].append(row)
                        for symmetric_row in symmetric_rows:
                            symmetric_subsets[symmetric_row[col]].append(symmetric_row)

                # Convert symmetric subsets back to DataFrames
                symmetric_subsets = {
                    value: pd.DataFrame(rows, columns=config_data.columns) for value, rows in
                    symmetric_subsets.items()
                    if rows
                }

                # Ensure symmetry (all subsets must have the same size)
                subset_sizes = [len(subset) for subset in symmetric_subsets.values()]
                if not all(size == subset_sizes[0] for size in subset_sizes):
                    print(f"Skipping dimension {col} for workload {workload} due to asymmetry.")
                    continue

                # Calculate FDC for each symmetric subset
                for value, subset in symmetric_subsets.items():
                    subset_indices = subset.index
                    subset_perf_data = perf_data[subset_indices]
                    subset_config_data = config_data.loc[subset_indices]

                    # Skip if the subset is too small
                    # if len(subset_perf_data) < len(perf_data) * 0.0001:
                    # continue

                    # Calculate FDC for the subset
                    ruggedness_value = self.calculate_autocorrelation_for_subset(subset_config_data, subset_perf_data)

                    row_result[f"{col}_{value}"] = ruggedness_value  # Add the FDC value as a new column

            sensitivity_results.append(row_result)  # Append the row result

        # Save the results to a CSV file
        sensitivity_df = pd.DataFrame(sensitivity_results)
        output_file = os.path.join(self.paths['landscape_metrics_AC'],
                                   f'{self.system_name}_ac_ind_sensitivity.csv')
        sensitivity_df.to_csv(output_file, index=False)
        print(f"individual option sensitivity of auto-correlation are saved to {output_file}")

    def calculate_autocorrelation_for_subset(self, config_data, performances):
        """
        Runs the full process for all workloads in the system:
        - Compute Hamming distances
        - Traverse the landscape
        - Compute autocorrelation for each workload
        """

        # Compute Hamming distances for this workload
        hamming_distances = self.compute_hamming_distances(config_data)

        # Traverse the landscape, generating the sequence of configurations
        sequence = self.traverse_landscape(hamming_distances, config_data)

        # Calculate autocorrelation based on the traversal sequence
        auto_correlation_value = self.autocorrelation_analysis(sequence, performances.tolist())

        return auto_correlation_value








