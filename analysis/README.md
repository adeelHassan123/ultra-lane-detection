# `analysis/` — Post-Training Evaluation & Visualization

This folder contains the suite of tools used to transform raw training logs and model checkpoints into the rich visualizations and metrics found in the project report. It serves as the **Evidence & Proof** layer of the research.

## 📂 Folder Structure

- **`ablation_chart.py`**: Generates a horizontal bar chart (with error bars) comparing all experiments (E0–E9). It visually "proves" which architectural changes had the most impact.
- **`confusion_matrix.py`**: Computes pixel-level classification accuracy (Lane vs. Background). It provides evidence of the model's precision at the pixel level, not just the image level.
- **`efficiency_plot.py`**: A scatter plot comparing **Model Accuracy (IoU)** vs. **Model Size (Parameters)**. This provides proof of the efficiency gains (e.g., how much more "bang for the buck" E7 provides compared to E6).
- **`error_analysis.py`**: The "Gallery" generator. It ranks validation predictions by IoU and creates grids of the **Best (Successes)** and **Worst (Failures)** results. This is crucial for qualitative evidence.
- **`plot_curves.py`**: Standard training/validation curves for Loss and IoU. It proves model convergence and shows whether a model was overfit (gap between train/val curves).
- **`results_table.py`**: Aggregates all 3-seed averages into a master CSV and a formatted PNG table. This is the "Gold Standard" of evidence for the project.

## 🚀 Key Workflows

### 1. Generating the Master Report
Once all experiments are complete, running `python scripts/run_analysis.py` triggers every script in this folder to:
1. Read logs from `outputs/logs/`.
2. Load checkpoints from `outputs/checkpoints/`.
3. Save high-resolution PNGs to `outputs/figures/`.

### 2. The "Ablation" Proof
The `ablation_chart.py` is specifically designed to show the "Delta" (change) in performance. By comparing E0 (Baseline) to E9 (Ablation), we can mathematically prove how much value "Data Augmentation" adds vs. "Architecture".

## 📊 Outputs (Proof Artifacts)

| Artifact | File Name | Purpose |
| :--- | :--- | :--- |
| **Comparative Chart** | `ablation_chart.png` | Ranks all 10 experiments by Mean IoU. |
| **Efficiency Graph** | `efficiency_plot.png` | Proves that bigger models (U-Net) are worth the extra parameters. |
| **Visual Gallery** | `failures_e7_transfer.png` | Shows exactly where the model struggles (e.g., sharp curves). |
| **Pixel Accuracy** | `confusion_summary.png` | Breaks down False Positives vs. False Negatives. |

## 🛠️ Internal Mechanics
Most scripts here use a standard pattern:
1. **Load Data**: Read `.csv` or `.pth` files using `pandas` or `torch`.
2. **Process**: Average metrics across 3 seeds (`42, 123, 777`).
3. **Visualize**: Use `matplotlib` with a custom color palette (Blue for CNNs, Green for U-Nets).
