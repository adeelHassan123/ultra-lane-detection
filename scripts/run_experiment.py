import argparse
from dataclasses import replace

from configs.cnn_config import CNNConfig
from configs.unet_config import UNetConfig
from experiments import run_experiment
from utils.logger import ResultsLogger


def get_cfg(exp_name: str):
    if exp_name == "e0":
        return CNNConfig(experiment_name="e0_baseline")
    if exp_name == "e1":
        return replace(CNNConfig(experiment_name="e1_optimizer"), optimizer_name="adam")
    if exp_name == "e2":
        return replace(CNNConfig(experiment_name="e2_activation"), activation="gelu")
    if exp_name == "e3":
        return CNNConfig(experiment_name="e3_regularization")
    if exp_name == "e4":
        return replace(CNNConfig(experiment_name="e4_augmentation"), train_augmentation="heavy")
    if exp_name == "e5":
        return replace(CNNConfig(experiment_name="e5_batchnorm"), use_batchnorm=True)
    if exp_name == "e6":
        return replace(UNetConfig(experiment_name="e6_architecture"), pretrained=False)
    if exp_name == "e7":
        return replace(UNetConfig(experiment_name="e7_transfer"), pretrained=True)
    if exp_name == "e8":
        return replace(UNetConfig(experiment_name="e8_loss"), loss_name="dice_bce")
    if exp_name == "e9":
        return replace(UNetConfig(experiment_name="e9_ablation"), pretrained=True, train_augmentation="none")
    raise ValueError(f"Unknown experiment: {exp_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, help="e0..e9")
    args = parser.parse_args()
    cfg = get_cfg(args.exp)
    summary = run_experiment(cfg)
    print(summary)
    result_row = {
        "experiment": cfg.experiment_name,
        "mean_iou": summary["mean_iou"],
        "std_iou": summary["std_iou"],
    }
    result_row["scores"] = summary["scores"]
    ResultsLogger(f"{cfg.results_dir}/master_results.csv").log(result_row)


if __name__ == "__main__":
    main()
