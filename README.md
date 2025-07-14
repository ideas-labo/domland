# Revealing Domain-Spatiality Patterns for Configuration Tuning: Domain Knowledge Meets Fitness Landscapes

# 1 Documents

## 1.1 datasets
The `dataset` folder contains configuration datasets for the following **9 subject systems**, each stored in a separate subfolder. The name of each subfolder corresponds to the system it represents.

| System   | Language | Domain          | Performance | Version      | Resource Intensity       | LOC     | #O  | #C   | #W  | Optimization Goal |
|----------|---------|------------------|-------------|--------------|--------------------------|---------|----|-------|-----|-------------------|
| JUMP3R   | Java    | Audio Encoder    | Runtime     | 1.0.4        | Computing                | 29,685  | 13 | 4,196 | 6   | Minimization      |
| KANZI    | Java    | File Compressor  | Runtime     | 1.9          | I/O; Memory              | 28,614  | 18 | 4,112 | 9   | Minimization      |
| DCONVERT | Java    | Image Scaling    | Runtime     | 1.0.0-α7     | I/O; Memory              | 6,888   | 10 | 6,764 | 12  | Minimization      |
| H2       | Java    | Database         | Throughput  | 4.200        | I/O; Memory; Storage     | 333,539 | 16 | 1,954 | 8   | Maximization      |
| BATLIK   | Java    | SVG Rasterizer   | Runtime     | 1.14         | Computing                | 360,924 | 8 | 1,919 | 11  | Minimization      |
| XZ       | C/C++   | File Compressor  | Runtime     | 5.2.0        | Computing                | 43,130  | 12 | 1,999 | 13  | Minimization      |
| LRZIP    | C/C++   | File Compressor  | Runtime     | 0.651        | Computing                | 20,797  | 7 | 190   | 13  | Minimization      |
| X264     | C/C++   | Video Encoder    | Runtime     | baee400...   | Computing; I/O           | 86,740  | 25 | 3,113 | 9   | Minimization      |
| Z3       | C/C++   | SMT Solver       | Runtime     | 4.8.14       | Computing                | 636,268 | 12 | 1,011 | 12  | Minimization      |

- **#O**: Number of options.
- **#C**: Number of configurations.
- **#W**: Number of workloads tested.
- **Optimization Goal**: Whether the goal is **minimization** (e.g., reducing runtime) or **maximization** (e.g., increasing throughput).

Each system folder contains **6 ~ 13 workloads**, depending on the system. The data is stored in `.csv` format, where each CSV file follows the below structure:

- **Columns (1 to n-1):** Configuration options, which can be either discrete or continuous values.
- **Column n:** Performance objective, represented as a numeric value (e.g., runtime, throughput).


## 1.2 landscape_results

The `landscape_results/` directory contains the **landscape analysis results** for each system. Each system's results are stored in a dedicated subfolder, and the results are further divided into **two categories**:

1. **`landscape_metrics/`**: Stores the **quantitative** results of various landscape analysis metrics.
2. **`landscape_visualizations/`**: Stores the **visual representations** of the landscape characteristics.

### 1.2.1 Directory Structure

```
landscape_results/
│── <system_name>/                    # Each system has its own folder
│   ├── landscape_metrics/            # Quantitative results of landscape metrics
│   │   ├── AC/                      # Autocorrelation results
│   │   ├── BoA/                     # Basin of Attraction results
│   │   ├── FDC/                     # Fitness Distance Correlation results
│   │   ├── local_structure.csv      # CSV summarizing local landscape structure
│   ├── landscape_visualizations/    # Visual representations of landscape properties
│   │   ├── basin_pics/              # Basins of attraction visualization
│   │   ├── full_landscape/          # Full landscape visualization
│   │   ├── local_optima/            # Local optima distribution visualization
```         

### 1.2.2 **Landscape Metrics (`landscape_metrics/`)**
The **`landscape_metrics/`** folder contains different **quantitative landscape metrics** computed for each system:

- **`AC/` (Autocorrelation)**: Measures the ruggedness of the landscape.
- **`BoA/` (Basin of Attraction)**: Analyzes the attraction basins of global optimum or local optima.
- **`FDC/` (Fitness Distance Correlation)**: Evaluates the correlation between fitness values and distances to the global optimum.
- **`local_structure.csv`**: A CSV file summarizing the local landscape structure for the system.

### 1.2.3 **Landscape Visualizations (`landscape_visualizations/`)**
The **`landscape_visualizations/`** folder contains **graphical representations** of the system's fitness landscape:

- **`basin_pics/`**: Visual the relationship between the performance of local optima and their basin size.
- **`full_landscape/`**: 3D visualizations of the complete fitness landscape.
- **`local_optima/`**: Visualizations of local optima distributions.

Each system folder in `landscape_results/` follows this structure, enabling both **quantitative** and **visual** analysis of landscape characteristics.


## 1.3 landscape_summary

The `landscape_summary/` directory contains summarized results of landscape analysis, making it easier to present findings in the paper. These tables aggregate key **quantitative metrics** extracted from `landscape_results/`, facilitating comparisons across different systems and workloads.

## 1.4 supplementary_file

The `supplementary/` directory contains supplementary materials that provide detailed introduction into the **case study** and the **landscape metrics** used in the paper.

```
supplementary_file/
│── example_visualization/                      # Example visualization (Figure 2)
│── option_features_details/        # Option details for each system in the case study
│   ├── Batik.csv
│   ├── Dconvert.csv
│   ├── H2.csv
│   ├── Jump3r.csv
│   ├── Kanzi.csv
│   ├── Lrzip.csv
│   ├── X264.csv
│   ├── Xz.csv
│   ├── Z3.csv
│── APPENDIX.pdf                                # Appendix describing landscape metrics in detail
│── option_features_summary.csv     # Summary of all system option features in the case study
│── screening_keywords.csv          # Keywords used for classifying option features
│── workload_features_summary.csv   # Summary of all system workload features in the case study
```

- **`option_features_details/`**:
This folder contains CSV files where each file corresponds to a system in the case study, listing its configuration options, descriptions, and type.

- **`APPENDIX.pdf`**:
A supplementary document that provides detailed explanations of the landscape metrics** used in the analysis.

- **`option_features_summary.csv`**
A summary of all system option features analyzed in the case study.

- **`screening_keywords.csv`**
Contains **keywords** used for classifying different option features, helping to standardize categorization.

- **`workload_features_summary.csv`**
A summary of workload features for all systems in the case study, providing an overview of workload characteristics across different systems.

This supplementary data ensures transparency and reproducibility for the analysis conducted in the study.

## 1.5 Validation
The `validation/` directory contains the experimental results that validate the actionable insights proposed in Section 5.5 of our paper. This section evaluates how different landscape characteristics influence the performance of configuration tuning algorithms, based on the following five insights:

- `insight1/`: Validates that ruggedness-sensitive options impact tuning efficiency and prioritization can benefit tuning algorithm.
- `insight234/`: Validates that landscape structure (e.g., smoothness, modality) affects the effectiveness of exploration-, exploitation-, and model-based tuners.
- `insight5/`: Examines the transferability of configurations across workloads under different degrees of landscape stability.

Each folder contains experimental data, evaluation scripts, and plotting functions used to support the corresponding insight in the paper.


# 2 Code Structure

The repository contains several Python scripts that are used for analyzing configuration landscapes, computing metrics, and generating visualizations. Below is an overview of each script and its functionality.


## 2.1 Main Python Files

```
│── landscape.py           # Core script for computing landscape metrics
│── main.py                # Entry point for running the full analysis pipeline
│── statistical_tool.py     # Statistical analysis functions for comparing landscapes
│── utils.py               # Utility functions for data processing and transformation
│── visualization.py       # Functions for generating landscape visualizations
```

### 2.1.1 **`landscape.py`**
This script is responsible for computing key **landscape analysis metrics**, such as:

- **Fitness Distance Correlation (FDC)**
- **Autocorrelation (AC)**
- **Basin of Attraction (BoA)**
- **Local optima detection and structure analysis (number/quality of local optima)**

It loads configuration-performance data from `datasets/`, computes landscape properties, and stores results in `landscape_results/`.

### 2.1.2 **`main.py`**
This script acts as the **entry point** for executing the full landscape analysis pipeline. It:

- Calls `landscape.py` to compute metrics
- Uses `visualization.py` to generate landscape visualizations
- Can be configured to run for specific systems

Ensure you are using **Python 3.9** and install the required dependencies:  
```bash
pip install -r requirements.txt
```

Then, run the analysis:  
```bash
python main.py
```

### 2.1.3 **`statistical_tool.py`**
This script is responsible for **summarizing computed landscape metrics** and **storing results in `landscape_tables/`**.  
It includes functions for:

- **Summarizing Fitness Distance Correlation (FDC)** → `summarize_fdc()`
- **Summarizing local landscape structure** → `summarize_local_structure()`
- **Summarizing global basin properties** → `summarize_global_basin()`
- **Summarizing autocorrelation** → `summarize_autocorrelation()`
- **Analyzing the relationship between basins and performance** → `basin_vs_perf_spearman()`
- **Autocorrelation sensitivity analysis** → `summarize_autocorrelation_sensitivity()`

These summaries are exported as CSV tables in `landscape_tables/` to facilitate research paper analysis.

### 2.1.4 **`utils.py`**
This script is responsible for **loading and preprocessing datasets** before analysis.  

### 2.1.5 **`visualization.py`**
This script is responsible for **generating visual representations** of the landscape analysis results. It supports:

- **Fitness landscapes (3D)**
- **Basin of attraction visualizations**
- **Local optima distribution plots**

Generated plots are saved under `landscape_results/landscape_visualizations/`.


