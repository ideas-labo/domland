import os
import pandas as pd
import re
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr


class Utils:
    def __init__(self, system_name):
        self.system_path = os.path.join('datasets', system_name)
        self.system_name = system_name
        self.perf_dict = self.load_perf_data(self.system_path)
        self.config_dict = self.load_config_data(self.system_path)

        # Define all required directories
        self.paths = {
            "landscape_metrics": ('landscape_results', system_name, 'landscape_metrics'),
            "landscape_metrics_FDC": ('landscape_results', system_name, 'landscape_metrics', 'FDC'),
            "landscape_metrics_BoA": ('landscape_results', system_name, 'landscape_metrics', 'BoA'),
            "landscape_metrics_AC": ('landscape_results', system_name, 'landscape_metrics', 'AC'),
        }

        # Generate full path strings
        self.paths = {key: os.path.join(*value) for key, value in self.paths.items()}

        # Create directories if they do not exist
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
    @staticmethod
    def extract_parts(file_name):
        """Extract the letters and numbers from file name"""
        match = re.match(r'([a-zA-Z]+)(\d+)', file_name)
        if match:
            return match.group(1), int(match.group(2))
        else:
            return file_name, 0

    @staticmethod
    def load_perf_data(system_path):
        """get the performance data of all workloads"""
        performance_dict = {}
        file_names = sorted([f for f in os.listdir(system_path) if f.endswith('.csv') and '_2d' not in f and '_FDC' not in f], key=Utils.extract_parts)

        for file_name in file_names:
            file_path = os.path.join(system_path, file_name)
            df = pd.read_csv(file_path)
            workload_name = file_name.split('.')[0]
            performance_dict[workload_name] = df.iloc[:, -1]  # performance is stored at the last column.
        return performance_dict

    @staticmethod
    def load_config_data(system_path):
        """get the configuration data of all workloads"""
        config_dict = {}
        file_names = sorted([f for f in os.listdir(system_path) if f.endswith('.csv') and '_2d' not in f and '_FDC' not in f], key=Utils.extract_parts)

        for file_name in file_names:
            file_path = os.path.join(system_path, file_name)
            df = pd.read_csv(file_path)
            workload_name = file_name.split('.')[0]
            config_dict[workload_name] = df.iloc[:, :-1]  # configuration is stored at first n-1 column
        return config_dict

    def calculate_spearman_cor_basin_vs_perf(self, basin_size_dict):
        """
        Compute the Spearman correlation between local optima performance and basin size for each workload.
        If multiple local optima have the same performance, their basin sizes are merged (summed) before computing correlation.

        :param system_name: Name of the system being analyzed.
        :param basin_size_dict: Dictionary containing basin sizes for each workload.
        :param perf_dict: Dictionary mapping workload to performance values.
        :param output_path: Directory where the CSV file should be saved.
        :return: DataFrame with workloads and their Spearman correlation values.
        """
        correlation_data = []

        for workload, basin_info in basin_size_dict.items():
            local_optima_indices = list(map(int, basin_info.keys()))
            basin_sizes = list(basin_info.values())

            optima_performances = [self.perf_dict[workload][idx] for idx in local_optima_indices]

            # Adjust performance values for minimization problems (invert sign)
            if self.system_name != 'h2':
                optima_performances = [-p for p in optima_performances]

            # Merge basin sizes for duplicate performance values
            merged_data = {}
            for perf, basin in zip(optima_performances, basin_sizes):
                if perf in merged_data:
                    merged_data[perf] += basin  # Sum basin sizes for identical performance values
                else:
                    merged_data[perf] = basin

            # Extract merged values for correlation computation
            merged_performances = list(merged_data.keys())
            merged_basins = list(merged_data.values())

            # Compute Spearman correlation
            if len(merged_performances) > 1 and len(merged_basins) > 1:
                correlation, _ = spearmanr(merged_performances, merged_basins)
            else:
                correlation = np.nan  # Not enough data points for correlation

            correlation_data.append([workload, correlation])

        # Save results as a CSV file
        df = pd.DataFrame(correlation_data, columns=['Workload', 'Spearman_Correlation'])
        csv_file = os.path.join(self.paths["landscape_metrics_BoA"], f'{self.system_name}_spearman_cor_basin_vs_perf.csv')
        df.to_csv(csv_file, index=False)

        return df


