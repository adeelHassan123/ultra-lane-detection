from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

@dataclass
class BaseConfig:
    experiment_name: str = "base"
    seed: int = 42
    seeds: List[int] = field(default_factory=lambda: [42, 123, 777])
    image_size: int = 256
    num_workers: int = 2
    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 1e-3
    encoder_learning_rate: float = 1e-4
    decoder_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    threshold: float = 0.5
    amp: bool = True

    data_root: str = "processed_data"

    outputs_root: str = "outputs"
    checkpoints_dir: str = "outputs/checkpoints"
    logs_dir: str = "outputs/logs"
    figures_dir: str = "outputs/figures"
    results_dir: str = "outputs/results"

    model_name: str = "baseline_cnn"
    optimizer_name: str = "adamw"
    loss_name: str = "combined"
    scheduler_name: str = "cosine"
    pretrained: bool = False
    encoder_name: str = "resnet34"
    dropout: float = 0.1
    use_batchnorm: bool = True
    activation: str = "relu"

    train_augmentation: str = "light"
    val_augmentation: str = "none"

    save_every: int = 5
    clip_grad_norm: float = 1.0
    positive_weight: float = 3.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    tversky_smooth: float = 1e-6
    tversky_weight: float = 0.9
    freeze_encoder_epochs: int = 15

    class_names: Tuple[str, str] = field(default_factory=lambda: ("background", "lane"))

    def ensure_output_dirs(self) -> None:
        for directory in (
            self.outputs_root,
            self.checkpoints_dir,
            self.logs_dir,
            self.figures_dir,
            self.results_dir,
        ):
            Path(directory).mkdir(parents=True, exist_ok=True)

    @property
    def train_images_dir(self) -> str:
        return str(Path(self.data_root) / "train" / "images")

    @property
    def train_masks_dir(self) -> str:
        return str(Path(self.data_root) / "train" / "masks")

    @property
    def val_images_dir(self) -> str:
        return str(Path(self.data_root) / "val" / "images")

    @property
    def val_masks_dir(self) -> str:
        return str(Path(self.data_root) / "val" / "masks")