import os
import re
import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances
import seaborn as sns
from utils import Utils
import pandas as pd
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
import random
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.interpolate import Rbf


class Visualization:
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
            "full_landscape": ('landscape_results', system_name, 'landscape_visualizations', 'full_landscape'),
            "local_optima": ('landscape_results', system_name, 'landscape_visualizations', 'local_optima'),
            "landscape_basin_pics": ('landscape_results', system_name, 'landscape_visualizations', 'basin_pics')
            # "heatmap": ('landscape_results', system_name, 'landscape_visualizations', 'heatmap')
        }

        # Generate full path strings
        self.paths = {key: os.path.join(*value) for key, value in self.paths.items()}

        # Create directories if they do not exist
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)

    def transform_configs_by_mds(self, n_components=2):
        """Conduct MDS reduction on all workloads for target system"""
        config_2d_dict = {}
        for workload, config_data in self.config_dict.items():
            file_path = os.path.join(self.system_path_2d, f"{workload}_2d.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                config_2d_dict[workload] = df[['MDS_Dim1', 'MDS_Dim2']].values
            else:
                # if config_2d_dict exist, retrieve it for saving time
                # as the config space is the same for all workload under one system, only get once
                if config_2d_dict:
                    config_2d = list(config_2d_dict.values())[0]
                    config_2d_dict[workload] = config_2d
                else:
                    # Normalization
                    config_data = pd.DataFrame(config_data)
                    preprocessed_data = self.preprocess_data(config_data)
                    mds = MDS(n_components=n_components, random_state=0)
                    config_2d = mds.fit_transform(preprocessed_data)
                    config_2d_dict[workload] = config_2d

                # logging the reduced data to csv
                df_2d = pd.DataFrame(config_2d, columns=['MDS_Dim1', 'MDS_Dim2'])
                df_2d['performance'] = self.perf_dict[workload]
                df_2d.to_csv(file_path, index=False)

        return config_2d_dict

    @staticmethod
    def preprocess_data(config_data):
        """perform normalization for non-binary variables"""
        # find none binary columns
        non_binary_columns = [col for col in config_data.columns if not all(config_data[col].isin([0, 1]))]
        if len(non_binary_columns) > 0:
            scaler = MinMaxScaler()
            config_data[non_binary_columns] = scaler.fit_transform(config_data[non_binary_columns])
        return config_data

    def full_landscape_visualization(self, config_2d_dict):
        """
        Visualize the complete landscape of each workload, marking all global optima.
        :param config_2d_dict: 2D configurations
        """
        for workload, config_2d in config_2d_dict.items():
            performance = self.perf_dict[workload]

            # Find global optima based on system name
            if self.system_name == 'h2':
                # For h2, higher performance is better (maximization)
                global_opt_indices = np.where(performance == np.max(performance))[0]
            else:
                # For other systems, lower performance is better (minimization)
                global_opt_indices = np.where(performance == np.min(performance))[0]


            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            # Scatter plot for the whole landscape
            sc = ax.scatter(config_2d[:, 0], config_2d[:, 1], performance,
                            c=performance, cmap='Spectral', marker='o', alpha=0.8)

            # Add color bar
            plt.subplots_adjust(right=0.85)
            if self.system_name in ['h2', 'a_redis']:
                cbar = plt.colorbar(sc, ax=ax, shrink=0.6, aspect=10, pad=0.02)
                cbar.set_label('Throughput', fontsize=18)
            else:
                cbar = plt.colorbar(sc, ax=ax, shrink=0.6, aspect=10, pad=0.02)
                cbar.set_label('Runtime', fontsize=18)

            # Mark each global optimum with a black pentagram
            for global_opt_index in global_opt_indices:
                global_opt_config = config_2d[global_opt_index]
                global_opt_perf = performance[global_opt_index]
                ax.scatter(global_opt_config[0], global_opt_config[1], global_opt_perf,
                           color='black', marker='*', s=150, edgecolors='white', linewidths=1.5,
                           label='Global Optimum' if global_opt_index == global_opt_indices[0] else "")

            # Improve axis labels and adjust their positions
            ax.set_xlabel('#D1', fontsize=16, labelpad=30)
            ax.set_ylabel('#D2', fontsize=16, labelpad=30)
            # ax.set_zlabel('Runtime', fontsize=16, labelpad=20)
            # plt.show()
            # Adjust view angle for better visibility
            ax.view_init(elev=35, azim=220)

            # Save figure
            file_path = os.path.join(self.paths["full_landscape"], f'{workload}_landscape.pdf')
            plt.savefig(file_path, bbox_inches='tight', format='pdf')

            plt.close()

    def full_landscape_visualization_3d_normalization(self, config_2d_dict):
        """
        Visualize the complete landscape of each workload as a 3D surface with normalized performance.
        :param config_2d_dict: 2D configurations
        """
        for workload, config_2d in config_2d_dict.items():
            performance = self.perf_dict[workload]


            if self.system_name in ['h2', 'a_redis']:
                global_opt_indices = np.where(performance == np.max(performance))[0]  # Maximization
            else:
                global_opt_indices = np.where(performance == np.min(performance))[0]  # Minimization
                performance = -performance  # Invert performance for minimization problems


            x = config_2d[:, 0]
            y = config_2d[:, 1]
            z = performance

            # insert data
            grid_x, grid_y = np.meshgrid(
                np.linspace(x.min(), x.max(), 20),  # resolution
                np.linspace(y.min(), y.max(), 20)
            )
            grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')  # nearest linear cubic
            # rbf_func = Rbf(x, y, z, function='cubic', smooth=0.3)
            # grid_z = rbf_func(grid_x, grid_y)

            # normalization
            z_min, z_max = np.nanmin(grid_z), np.nanmax(grid_z)
            if z_max > z_min:
                grid_z = (grid_z - z_min) / (z_max - z_min)
            else:
                grid_z = np.zeros_like(grid_z)


            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='Spectral', edgecolor='k', linewidth=0.3, alpha=0.95, vmin=0, vmax=0.8)

            # grid
            # ax.plot_wireframe(grid_x, grid_y, grid_z, color='black', linewidth=0.2, alpha=0.5)

            cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.02)
            # cbar.set_label('Normalized Performance', fontsize=18)

            if self.system_name in ['h2', 'a_redis']:

                cbar.set_label('Normalized Throughput', fontsize=22)
            else:

                cbar.set_label('Normalized Negative Runtime', fontsize=22)


            # for global_opt_index in global_opt_indices:
            #     ax.scatter(x[global_opt_index], y[global_opt_index], grid_z.flatten()[global_opt_index],
            #                color='black', marker='*', s=200, edgecolors='white', linewidths=1.5,
            #                label='Global Optimum' if global_opt_index == global_opt_indices[0] else "")

            ax.set_xlabel('#D1', fontsize=22, labelpad=15)
            ax.set_ylabel('#D2', fontsize=22, labelpad=15)
            # ax.set_zlabel('Normalized Performance', fontsize=16)
            # plt.show()
            ax.view_init(elev=35, azim=220)

            file_path = os.path.join(self.paths["full_landscape"], f'{workload}_3D_normalized.pdf')
            plt.savefig(file_path, bbox_inches='tight', format='pdf')
            plt.close()

    def basin_vs_perf_visualization(self, basin_size_dict):
        """
        Use bubble plots to visualize the performance of the local optimum under each workload and the size of
        its basin of attraction.
        1. Horizontal coordinate: the performance of local optima
        2. Vertical coordinate: the size of basin of local optima
        3. Bubble size: map to basin size for enhanced visualization
        """
        for workload, basin_info in basin_size_dict.items():
            performances = self.perf_dict[workload]
            local_optima_indices = list(basin_info.keys())
            basin_sizes = list(basin_info.values())

            optima_performances = [performances[int(idx)] for idx in local_optima_indices]

            # Identify global optima
            if self.system_name == 'h2':
                global_optima_indices = np.where(performances == np.max(performances))[0]
            else:
                global_optima_indices = np.where(performances == np.min(performances))[0]

            plt.figure(figsize=(8, 6))

            # Plot local optima (as bubbles)
            plt.scatter(optima_performances, basin_sizes,
                        s=[size * 2 for size in basin_sizes],
                        c=optima_performances, cmap='RdBu', alpha=0.6, edgecolors='black')

            # plt.title(f'The basin size and performance of local optima')

            if self.system_name == 'h2':
                plt.xlabel('Throughput', fontsize=20)
                plt.xticks(fontsize=18)
                cbar = plt.colorbar()
                cbar.set_label('Throughput', fontsize=20)  # colorbar label font size
                cbar.ax.tick_params(labelsize=18)
            else:
                plt.xlabel('Runtime', fontsize=20)
                plt.xticks(fontsize=18)
                # show bar
                cbar = plt.colorbar()
                cbar.set_label('Runtime', fontsize=18)
                cbar.ax.tick_params(labelsize=18)


            plt.ylabel('Basin size', fontsize=18)
            plt.yticks(fontsize=18)



            # Highlight global optima with stars
            for idx in global_optima_indices:
                plt.scatter(performances[idx], basin_info[str(idx)],
                            s=basin_info[str(idx)] * 2,
                            color='white', marker='*', edgecolors='white',
                            label='Global Optimum')


            plt.tight_layout()
            file_path = os.path.join(self.paths["landscape_basin_pics"], f'{workload}_basin_vs_perf.pdf')
            plt.savefig(file_path, bbox_inches='tight', format='pdf')
            plt.close()
            # plt.show()

    def local_optima_visualization(self, workload, local_optima_dict, config_2d_dict):
        """
        visualize the local optima
        :param workload: the name of workload
        :param local_optima_dict:
        :param config_2d_dict:
        """
        config_2d = config_2d_dict[workload]
        performance = self.perf_dict[workload]

        # Get the indices of local optimal configs
        local_optima_indices = local_optima_dict[workload]

        if self.system_name in ['h2', 'a_redis']:
            sorted_optima = sorted(local_optima_indices, key=lambda x: performance[x], reverse=True)
        else:
            sorted_optima = sorted(local_optima_indices, key=lambda x: performance[x])

        # Get the performances according to the indices of local optima
        optima_performances = np.array([performance[idx] for idx in sorted_optima])

        # Normalize the performance for mapping performance to color
        norm_performances = (optima_performances - optima_performances.min()) / (
                optima_performances.max() - optima_performances.min())

        # Set color and size according to performance
        # colors = cm.Spectral(norm_performances)
        sizes = 100 + norm_performances * 200  # Marker size based on performance

        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot each local optima with corresponding color and size
        scatter = ax.scatter(config_2d[sorted_optima, 0], config_2d[sorted_optima, 1],
                             c=optima_performances, s=sizes, cmap='Spectral', edgecolors='black', alpha=0.7)

        # Add color bar to show the performance values
        if self.system_name == 'h2':
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Throughput', fontsize=26)
        else:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Runtime', fontsize=26)

        ax.set_xlabel('#D1', fontsize=26)
        ax.set_ylabel('#D2', fontsize=26)
        # ax.set_title(f'Local Optima Distribution for W: ({workload}) of S: ({self.system_name})')
        file_path = os.path.join(self.paths["local_optima"], f'{workload}_local_optima.pdf')
        plt.savefig(file_path, bbox_inches='tight', format='pdf')
        # plt.show()
        plt.close()

    def local_optima_overlap_heatmap_visualization(self, local_optima_dict):
        """
        Calculate the Jaccard correlation between the local optima of paired workloads
        :param local_optima_dict: workload: local optima index
        """
        workloads = list(local_optima_dict.keys())
        num_workloads = len(workloads)


        jaccard_matrix = np.zeros((num_workloads, num_workloads))


        for i in range(num_workloads):
            for j in range(num_workloads):
                if i == j:
                    jaccard_matrix[i][j] = 1.0
                else:
                    optima_i = set(local_optima_dict[workloads[i]])
                    optima_j = set(local_optima_dict[workloads[j]])

                    # Compute intersection and union
                    intersection_count = len(optima_i & optima_j)
                    union_count = len(optima_i | optima_j)

                    # Calculate Jaccard correlation
                    jaccard_similarity = intersection_count / union_count if union_count > 0 else 0.0
                    jaccard_matrix[i][j] = jaccard_similarity

        plt.figure(figsize=(10, 8))
        sns.heatmap(jaccard_matrix, annot=True, xticklabels=workloads, yticklabels=workloads, cmap="BuGn", cbar=True)
        plt.title('Jaccard Similarity Between Workloads')
        plt.xlabel('Workloads')
        plt.ylabel('Workloads')

        file_path = os.path.join(self.paths["heatmap"], 'local_optima_jaccard_similarity.pdf')
        plt.savefig(file_path, bbox_inches='tight', format='pdf')
        # plt.show()
        plt.close()


