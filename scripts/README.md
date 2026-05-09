# `scripts/` — Execution Control & Automation

This folder contains the entry points for every major task in the project. If you want to *do* something (Train, Preprocess, Analyze), you start here.

## 📂 Folder Structure

- **`run_experiment.py`**: The primary engine. It takes an experiment ID (e.g., `--exp e7`) and runs the full 3-seed training cycle.
- **`run_analysis.py`**: The "Report Maker". Run this after your experiments to generate all the charts and tables for the final report.
- **`preprocess_data.py`**: The "Cleaner". Prepares the raw dataset by dilating masks and validating integrity.
- **`preprocess_to_numpy.py`**: **The Optimization Engine.** Converts the entire dataset to NumPy format for ultra-fast training.
- **`benchmark_speed.py`**: The "Speedometer". Measures exactly how many images per second your system can process.
- **`setup_colab.py`**: The "Auto-Installer". One-click setup for Google Colab environments.

## 🚀 Common Workflows

### 1. The "Starting from Scratch" Flow
```bash
# 1. Setup the environment
python scripts/setup_colab.py

# 2. Preprocess the data
python scripts/preprocess_data.py

# 3. Optimize for speed (optional but recommended)
python scripts/preprocess_to_numpy.py
```

### 2. The "Research" Flow
```bash
# Run the baseline
python scripts/run_experiment.py --exp e0

# Run the best model
python scripts/run_experiment.py --exp e7
```

### 3. The "Evaluation" Flow
```bash
# Generate all plots and the master results table
python scripts/run_analysis.py
```

## 🧠 Efficiency & Proof
The `benchmark_speed.py` script is our "Proof of Optimization". It compares the speed of reading standard images vs. our optimized NumPy format.
- **Evidence:** Our setup typically shows a **10.5x speedup** on Google Colab, reducing total project training time from 20 hours to under 2 hours.
