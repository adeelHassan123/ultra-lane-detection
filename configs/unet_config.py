from dataclasses import dataclass

from configs.base import BaseConfig


@dataclass
class UNetConfig(BaseConfig):
    experiment_name: str = "e6_architecture"
    model_name: str = "unet_smp"
    pretrained: bool = True
    batch_size: int = 8
    epochs: int = 45
    train_augmentation: str = "heavy"
