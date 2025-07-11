import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class StatisticalTool:
    def __init__(self, dataset_path, system_names, max_workloads=13):
        self.dataset_path = dataset_path
        self.system_names = system_names
        self.max_workloads = max_workloads
        self.output_path = "landscape_summary"
        os.makedirs(self.output_path, exist_ok=True)

    def summarize_fdc(self):
        all_data = {}

        # Iterating each system and load {system_name}_FDC.csv
        for system_name in self.system_names:
            file_path = os.path.join(self.dataset_path, system_name, "landscape_metrics", 'FDC', f"{system_name}_FDC.csv")

            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found, skipping {system_name}.")
                continue

            df = pd.read_csv(file_path, usecols=["workload", "FDC"])
            df["workload"] = [f"W{i + 1}" for i in range(len(df))]  # rename W1, W2, ...
            all_data[system_name] = df.set_index("workload")  # Indexed by workload

        # Creates the final summarized DataFrame, supporting up to 13 workloads.
        workloads_list = [f"W{i + 1}" for i in range(self.max_workloads)]
        summary_df = pd.DataFrame(index=workloads_list, columns=self.system_names)

        # Fill data and process N/A
        for system_name, df in all_data.items():
            summary_df[system_name] = df.reindex(workloads_list)["FDC"]

        summary_df = summary_df.astype(str).replace("nan", "N/A")

        median_values = {}
        iqr_values = {}

        for system_name in self.system_names:
            valid_data = pd.to_numeric(summary_df[system_name], errors='coerce').dropna()
            median_values[system_name] = round(valid_data.median(), 3) if not valid_data.empty else "N/A"
            iqr_values[system_name] = round(valid_data.quantile(0.75) - valid_data.quantile(0.25), 3) if not valid_data.empty else "N/A"

        summary_df.loc["Median"] = median_values
        summary_df.loc["IQR"] = iqr_values

        output_file = os.path.join(self.output_path, "TABLE_FDC.csv")
        summary_df.to_csv(output_file)
        print(f"FDC summary saved to {output_file}")

        return summary_df

    def summarize_local_structure(self):
        all_data = {}

        # Iterating each system and load local_structure.csv
        for system_name in self.system_names:
            file_path = os.path.join(self.dataset_path, system_name, "landscape_metrics", "local_structure.csv")

            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found, skipping {system_name}.")
                continue

            df = pd.read_csv(file_path, usecols=["workload", "local_optima_ratio", "quality"])
            df["workload"] = [f"W{i + 1}" for i in range(len(df))]
            all_data[system_name] = df.set_index("workload")

        # Generate column name, {sys_name}_R, {sys_name}_Q
        workloads_list = [f"W{i + 1}" for i in range(self.max_workloads)]
        columns = []
        for sys in self.system_names:
            columns.append(f"{sys}_R")
            columns.append(f"{sys}_Q")

        summary_df = pd.DataFrame(index=workloads_list, columns=columns)

        for system_name in self.system_names:
            if system_name in all_data:
                df = all_data[system_name]
                summary_df[f"{system_name}_R"] = df.reindex(workloads_list)["local_optima_ratio"]
                summary_df[f"{system_name}_Q"] = df.reindex(workloads_list)["quality"]

        summary_df = summary_df.astype(str).replace("nan", "N/A")

        median_values = {}
        iqr_values = {}

        for col in columns:
            valid_data = pd.to_numeric(summary_df[col], errors='coerce').dropna()
            median_values[col] = round(valid_data.median(), 3) if not valid_data.empty else "N/A"
            iqr_values[col] = round(valid_data.quantile(0.75) - valid_data.quantile(0.25), 3) if not valid_data.empty else "N/A"

        summary_df.loc["Median"] = median_values
        summary_df.loc["IQR"] = iqr_values

        output_file = os.path.join(self.output_path, "TABLE_Local_Optima.csv")
        summary_df.to_csv(output_file)
        print(f"Local structure summary saved to {output_file}")

        return summary_df

    def summarize_global_basin(self):
        all_data = {}

        for system_name in self.system_names:
            file_path = os.path.join(self.dataset_path, system_name, "landscape_metrics", 'BoA',
                                     f"{system_name}_global_basin_ratios.csv")

            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found, skipping {system_name}.")
                continue

            df = pd.read_csv(file_path, usecols=["workload", "num_global_optima", "global_basin_ratio"])
            df["workload"] = [f"W{i + 1}" for i in range(len(df))]
            all_data[system_name] = df.set_index("workload")

        workloads_list = [f"W{i + 1}" for i in range(self.max_workloads)]
        columns = []
        for sys in self.system_names:
            # columns.append(f"{sys}_NG")
            columns.append(f"{sys}_GBR")

        summary_df = pd.DataFrame(index=workloads_list, columns=columns)

        for system_name in self.system_names:
            if system_name in all_data:
                df = all_data[system_name]
                # summary_df[f"{system_name}_NG"] = df.reindex(workloads_list)["num_global_optima"]
                summary_df[f"{system_name}_GBR"] = df.reindex(workloads_list)["global_basin_ratio"]

        summary_df = summary_df.astype(str).replace("nan", "N/A")

        median_values = {}
        iqr_values = {}

        for col in columns:
            valid_data = pd.to_numeric(summary_df[col], errors='coerce').dropna()
            median_values[col] = round(valid_data.median(), 3) if not valid_data.empty else "N/A"
            iqr_values[col] = round(valid_data.quantile(0.75) - valid_data.quantile(0.25), 3) if not valid_data.empty else "N/A"

        summary_df.loc["Median"] = median_values
        summary_df.loc["IQR"] = iqr_values

        output_file = os.path.join(self.output_path, "TABLE_Global_Basin.csv")
        summary_df.to_csv(output_file)
        print(f"Global Basin summary saved to {output_file}")

        return summary_df

    def summarize_autocorrelation(self):
        all_data = {}

        for system_name in self.system_names:
            file_path = os.path.join(self.dataset_path, system_name, "landscape_metrics", 'AC',
                                     f"{system_name}_autocorrelation.csv")

            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found, skipping {system_name}.")
                continue

            df = pd.read_csv(file_path, usecols=["workload", "autocorrelation"])
            df["workload"] = [f"W{i + 1}" for i in range(len(df))]
            all_data[system_name] = df.set_index("workload")

        workloads_list = [f"W{i + 1}" for i in range(self.max_workloads)]
        summary_df = pd.DataFrame(index=workloads_list, columns=self.system_names)

        for system_name in self.system_names:
            if system_name in all_data:
                df = all_data[system_name]
                summary_df[system_name] = df.reindex(workloads_list)["autocorrelation"]

        summary_df = summary_df.astype(str).replace("nan", "N/A")

        median_values = {}
        iqr_values = {}

        for system_name in self.system_names:
            valid_data = pd.to_numeric(summary_df[system_name], errors='coerce').dropna()
            median_values[system_name] = round(valid_data.median(), 3) if not valid_data.empty else "N/A"
            iqr_values[system_name] = round(valid_data.quantile(0.75) - valid_data.quantile(0.25), 3) if not valid_data.empty else "N/A"

        summary_df.loc["Median"] = median_values
        summary_df.loc["IQR"] = iqr_values

        output_file = os.path.join(self.output_path, "TABLE_Auto_Correlation.csv")
        summary_df.to_csv(output_file)
        print(f"Autocorrelation summary saved to {output_file}")

        return summary_df

    @staticmethod
    def basin_vs_perf_spearman(max_workloads=13):
        """
        Reads multiple Spearman correlation CSV files, constructs a unified DataFrame with workloads as rows and
        systems as columns, fills missing workloads with 'N/A', and generates a heatmap visualization.

        :param csv_files: Dictionary where keys are system names and values are CSV file paths.
        :param output_path: Path to save the generated heatmap image.
        :param max_workloads: Maximum number of workloads (default is 13).
        :return: Path to the saved heatmap image.
        """

        csv_files = {
            "lrzip": "landscape_results/lrzip/landscape_metrics/BoA/lrzip_spearman_cor_basin_vs_perf.csv",
            "xz": "landscape_results/xz/landscape_metrics/BoA/xz_spearman_cor_basin_vs_perf.csv",
            "z3": "landscape_results/z3/landscape_metrics/BoA/z3_spearman_cor_basin_vs_perf.csv",
            "dconvert": "landscape_results/dconvert/landscape_metrics/BoA/dconvert_spearman_cor_basin_vs_perf.csv",
            "batik": "landscape_results/batik/landscape_metrics/BoA/batik_spearman_cor_basin_vs_perf.csv",
            "kanzi": "landscape_results/kanzi/landscape_metrics/BoA/kanzi_spearman_cor_basin_vs_perf.csv",
            "x264": "landscape_results/x264/landscape_metrics/BoA/x264_spearman_cor_basin_vs_perf.csv",
            "h2": "landscape_results/h2/landscape_metrics/BoA/h2_spearman_cor_basin_vs_perf.csv",
            "jump3r": "landscape_results/jump3r/landscape_metrics/BoA/jump3r_spearman_cor_basin_vs_perf.csv"
        }
        output_path = "landscape_tables"

        all_data = {}

        # Load each CSV file
        for system_name, file_path in csv_files.items():
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found, skipping {system_name}.")
                continue

            df = pd.read_csv(file_path, usecols=["Workload", "Spearman_Correlation"])
            df["Workload"] = [f"W{i + 1}" for i in range(len(df))]  # Rename workloads W1, W2, ...
            all_data[system_name] = df.set_index("Workload")  # Indexed by workload

        # Create a summary DataFrame with up to max_workloads (default: 13 workloads)
        workloads_list = [f"W{i + 1}" for i in range(max_workloads)]
        summary_df = pd.DataFrame(index=workloads_list, columns=csv_files.keys())

        # Fill data and handle missing values
        for system_name, df in all_data.items():
            summary_df[system_name] = df.reindex(workloads_list)["Spearman_Correlation"]

        summary_df = summary_df.astype(str).replace("nan", "N/A")  # Replace NaN with "N/A"

        # Convert valid numeric values back to float for visualization
        numeric_summary_df = summary_df.replace("N/A", np.nan).astype(float)

        # Prepare figure
        fig, ax = plt.subplots(figsize=(10, 9))

        # Get coordinate positions
        x_labels = numeric_summary_df.columns.tolist()
        y_labels = numeric_summary_df.index.tolist()
        x_ticks = np.arange(len(x_labels))
        y_ticks = np.arange(len(y_labels))

        # Create grid using scatter plot with fixed circle sizes
        x_pos, y_pos, colors, values = [], [], [], []
        for i, workload in enumerate(y_labels):
            for j, system in enumerate(x_labels):
                value = numeric_summary_df.loc[workload, system]
                if not np.isnan(value):
                    x_pos.append(j)
                    y_pos.append(i)
                    colors.append(value)
                    values.append(value)

        # Scatter plot (bubble heatmap with uniform-sized circles)
        scatter = ax.scatter(x_pos, y_pos, s=1000, c=colors, cmap="RdBu", alpha=0.75, edgecolors="black")

        # Add value annotations in black
        for i, txt in enumerate(values):
            ax.text(x_pos[i], y_pos[i], f"{txt:.2f}", ha='center', va='center', fontsize=12, color="black")

        # Set labels and formatting
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([label.upper() for label in x_labels], rotation=45, ha="right", fontsize=10)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=10)
        ax.invert_yaxis()  # Match heatmap layout (top W1 → bottom W13)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Add grid lines
        ax.set_xticks(np.arange(len(x_labels) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(y_labels) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)  # Hide minor ticks

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Spearman correlation", fontsize=14)

        # plt.title("Spearman correlation between the performance of local optima and their basin size", fontsize=18)
        plt.xlabel("System", fontsize=14)
        plt.ylabel("Workload", fontsize=14)

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # Save figure
        heatmap_path = os.path.join(output_path, "basin_vs_perf_spearman.pdf")
        # plt.show()
        plt.savefig(heatmap_path, bbox_inches="tight", format='pdf')

    def summarize_autucorrelation_sensitivity(self):
        for system_name in self.system_names:

            input_file = os.path.join(self.dataset_path, system_name, "landscape_metrics", 'AC',
                                      f"{system_name}_ac_ind_sensitivity.csv")
            output_file = os.path.join(self.dataset_path, system_name, "landscape_metrics", 'AC',
                                      f"{system_name}_ac_ind_rsd_results.csv")


            if not os.path.exists(input_file):
                raise FileNotFoundError(f"input file {input_file} does not exists，please check the paht！")

            sensitivity_df = pd.read_csv(input_file)

            # make sure the file contains 'system' and 'workload' column
            if 'system' not in sensitivity_df.columns or 'workload' not in sensitivity_df.columns:
                raise ValueError("input data must contains 'system' and 'workload' column！")

            # 只保留数值列（去除 system 和 workload）
            numeric_data = sensitivity_df.drop(columns=['system', 'workload'])

            # ========== calculate the median for each option (with each value) across different workloads==========
            median_per_option = numeric_data.median().to_frame(name='Median_Value')

            # ========== Compute RSD for each option ==========
            option_rsd = {}
            for col in numeric_data.columns:
                option_name = col.rsplit('_', 1)[0]  # Extract option name
                if option_name not in option_rsd:
                    option_rsd[option_name] = []
                option_rsd[option_name].append(median_per_option.loc[col, 'Median_Value'])

            # Compute RSD (Relative Standard Deviation)
            option_rsd_results = {
                option: (np.std(values) / abs(np.mean(values))) * 100 if np.mean(values) != 0 else np.nan
                for option, values in option_rsd.items()
            }

            # ========== Generate final results table ==========
            rsd_df = pd.DataFrame({
                'Option': option_rsd_results.keys(),
                'RSD (%)': option_rsd_results.values()  # Display in percentage form
            })

            # ========== Save the result to a CSV file ==========
            rsd_df.to_csv(output_file, index=False, float_format="%.3f")
            print(f"The RSD of each option for system {system_name} is saved to {output_file}")

            # # ========== calculate variance for the option ==========
            # option_variances = {}
            # for col in numeric_data.columns:
            #     option_name = col.rsplit('_', 1)[0]  # extract option name
            #     if option_name not in option_variances:
            #         option_variances[option_name] = []
            #     option_variances[option_name].append(median_per_option.loc[col, 'Median_Value'])
            #
            # # for option, values in option_variances.items():
            # #     variance = np.var(values)
            # #     option_variances[option] = variance
            #
            #
            # option_variance_results = {option: np.var(values) for option, values in option_variances.items()}
            #
            # # ========== Generate final results table ==========
            # variance_df = pd.DataFrame({
            #     'Option': option_variance_results.keys(),
            #     'Variance': option_variance_results.values()
            # })
            #
            # # ========== save the result to the CSV file ==========
            # variance_df.to_csv(output_file, index=False)
            # print(f"The variance of each option for system {system_name} are saved to {output_file}")


if __name__ == "__main__":
    dataset_path = "landscape_results"
    system_names = ['lrzip', 'xz', 'z3', 'dconvert', 'batik', 'kanzi', 'x264', 'h2', 'jump3r']

    statistical_tool = StatisticalTool(dataset_path, system_names)
    # fdc_summary = statistical_tool.summarize_fdc()
    # local_structure_summary = statistical_tool.summarize_local_structure()
    # global_basin_summary = statistical_tool.summarize_global_basin()
    # autocorrelation_summary = statistical_tool.summarize_autocorrelation()
    statistical_tool.basin_vs_perf_spearman()
    # statistical_tool.summarize_autucorrelation_sensitivity()





