from dataclasses import dataclass

from configs.base import BaseConfig


@dataclass
class CNNConfig(BaseConfig):
    experiment_name: str = "e0_baseline"
    model_name: str = "baseline_cnn"
    pretrained: bool = False
    batch_size: int = 16
