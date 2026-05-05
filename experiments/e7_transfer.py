"""
E7: Transfer Learning — U-Net with ImageNet Pretrained ResNet-34
================================================================
Variable changed:  pretrained → True  (vs False in E6)
Hypothesis: ResNet-34 pretrained on ImageNet already detects edges, gradients,
            and textures. Fine-tuning it for lane detection should be far more
            sample-efficient than training from scratch.
Training strategy (implemented in Trainer):
  - Epochs  1–15: Encoder frozen, only decoder trains (lr=1e-3)
  - Epochs 16–45: Differential LR — encoder (1e-4) vs decoder (1e-3)
Expected outcome: best IoU of all experiments (≥0.60), establishing the ceiling.
Architecture: U-Net with ResNet-34 encoder (ImageNet pretrained)
              ~24M parameters | 45 epochs | heavy augmentation
Baseline (E0) mean IoU: 0.4968
"""

from dataclasses import replace

from configs.unet_config import UNetConfig


def get_config():
    base = UNetConfig(experiment_name="e7_transfer")
    return replace(
        base,
        pretrained=True,                    # Changed: ImageNet pretrained encoder
        train_augmentation="heavy",         # Heavy aug to prevent overfitting
        freeze_encoder_epochs=15,           # Freeze encoder for warmup phase
        encoder_learning_rate=1e-4,         # Gentle fine-tuning for pretrained layers
        decoder_learning_rate=1e-3,         # Faster learning for new decoder layers
    )
