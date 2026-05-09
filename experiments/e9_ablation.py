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
  base_cfg = UNetConfig(experiment_name="e10_ultimate") # New experiment name
  
  return replace(
      base_cfg,
      # --- Architecture & Pretraining (from E7) ---
      model_name="unet_smp",
      encoder_name="resnet34",
      pretrained=True,
      
      # --- Training Params (Optimized) ---
      epochs=60,                      # Increased from 45
      batch_size=8,
      learning_rate=1e-3,             # Default U-Net LR
      encoder_learning_rate=1e-4,     # Differential LR
      decoder_learning_rate=1e-3,     # Differential LR
      weight_decay=1e-4,
      patience=15,                    # Increased patience for early stopping
      min_delta=0.001,                # Ignore minor fluctuations
      
      # --- Loss & Augmentation (Proven effective) ---
      loss_name="combined",
      train_augmentation="heavy",     # Essential for robustness
      val_augmentation="none",
)

