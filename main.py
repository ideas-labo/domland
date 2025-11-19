import os
from landscape import Landscape
from visualization import Visualization
from utils import Utils


def main():
    # system_names = ['batik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    system_names = ['a_Apache']
    # system_names = ['batik']

    for system_name in system_names:

        """-----------------landscape analysis (Metric)------------------------"""
        landscape = Landscape(system_name)

        # 1. Fitness Distance Correlation (FDC)
        landscape.calculate_fdc()

        # 2. Local Structure (number, quality, and basin of attraction of local optima)
        local_optima_dict, hamming_distance_dict = landscape.calculate_local_optima()
        quality_dict = landscape.calculate_local_optima_quality(local_optima_dict)
        basin_size_dict = landscape.calculate_basin(local_optima_dict, hamming_distance_dict)
        landscape.save_local_structure(local_optima_dict, basin_size_dict, quality_dict, hamming_distance_dict)

        # 3. Basin of Attraction of global optimum
        basin_size_dict = landscape.load_basin_data()
        landscape.calculate_global_basin(basin_size_dict)

        # 4. Spearman correlation between basin size and superiority of local optima
        # landscape.utils.calculate_spearman_cor_basin_vs_perf(basin_size_dict)

        # 5. Ruggedness (autocorrelation)
        # landscape.calculate_autocorrelation()

        """-----------------landscape analysis (Visualization)------------------------"""
        visualization = Visualization(system_name)

        # 1. full landscape
        # config_2d_dict = visualization.transform_configs_by_mds()
        # visualization.full_landscape_visualization(config_2d_dict)
        # visualization.full_landscape_visualization_3d_normalization(config_2d_dict)

        # 2. basin size vs performance (for local optima)
        # basin_size_dict = landscape.load_basin_data()
        # visualization.basin_vs_perf_visualization(basin_size_dict)
        # for workload in landscape.config_dict.keys():
        #     visualization.local_optima_visualization(workload, local_optima_dict, config_2d_dict)


        "------------------landscape sensitivity (Metric) ------------------"

        # landscape.calculate_ind_option_impact_on_autocorrelation()
        # landscape_sensitivity.visualize_ruggedness_sensitivity()



if __name__ == "__main__":
    main()
