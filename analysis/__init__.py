"""
Analysis package — generates all report figures from training logs.

Usage (after all experiments are complete):
    python scripts/run_analysis.py

Functions exposed:
    build_results_table()     → master_results.csv + formatted table PNG
    plot_all_curves()         → per-experiment training/val curve PNGs
    generate_failure_gallery()→ worst-prediction image grid PNGs
    build_confusion_matrix()  → pixel-level confusion matrix PNGs
    create_ablation_chart()   → bar chart comparing E0–E9 mean IoU
    create_efficiency_plot()  → IoU vs param-count scatter plot
"""

from analysis.ablation_chart import create_ablation_chart
from analysis.confusion_matrix import build_confusion_matrix
from analysis.efficiency_plot import create_efficiency_plot
from analysis.error_analysis import generate_failure_gallery
from analysis.plot_curves import plot_all_curves
from analysis.results_table import build_results_table

__all__ = [
    "build_results_table",
    "plot_all_curves",
    "generate_failure_gallery",
    "build_confusion_matrix",
    "create_ablation_chart",
    "create_efficiency_plot",
]
