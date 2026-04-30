from analysis.ablation_chart import create_ablation_chart
from analysis.confusion_matrix import build_confusion_matrix
from analysis.efficiency_plot import create_efficiency_plot
from analysis.error_analysis import generate_failure_gallery
from analysis.plot_curves import plot_all_curves
from analysis.results_table import build_results_table


def main():
    build_results_table()
    plot_all_curves()
    generate_failure_gallery()
    build_confusion_matrix()
    create_ablation_chart()
    create_efficiency_plot()
    print("Analysis artifacts generated.")


if __name__ == "__main__":
    main()
