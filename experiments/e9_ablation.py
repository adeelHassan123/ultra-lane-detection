"""
E9: Ablation Study — Best Model Without Augmentation
=====================================================
Purpose: Isolate the contribution of data augmentation to E7's final IoU.
         Trains the best configuration (pretrained U-Net + combined loss) but
         with NO training augmentation, revealing how much of E7's gain comes
         from augmentation vs architecture/pretrained weights alone.

Variables (vs E7):
  - train_augmentation: "none"  (vs "heavy" in E7)   ← the only change

Expected outcome:
  - Ablation IoU < E7 IoU  → confirms augmentation adds value
  - Ablation IoU > E0 IoU  → confirms architecture/pretrained weights add value
  - The gap (E7 - Ablation) quantifies augmentation's isolated contribution.

Architecture: U-Net with ResNet-34 encoder (ImageNet pretrained)
              45 epochs | NO training augmentation
Baseline (E0) mean IoU: 0.4968 | E7 target: ≥0.60
"""

from dataclasses import replace

from configs.unet_config import UNetConfig


def get_config():
    base = UNetConfig(experiment_name="e9_ablation")
    return replace(
        base,
        pretrained=True,
        train_augmentation="none",          # Changed: "heavy" → "none" (ablation)
        freeze_encoder_epochs=15,
        encoder_learning_rate=1e-4,
        decoder_learning_rate=1e-3,
    )
