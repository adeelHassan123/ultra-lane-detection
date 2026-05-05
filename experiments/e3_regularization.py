"""
E3: Regularization Study
=========================
Variable changed:  dropout + weight_decay  → higher values
Hypothesis: Stronger regularization should reduce overfitting on the ~2500-image
            training set, improving the gap between train and val IoU.
Config:
  - dropout: 0.3 (vs 0.1 in E0)   — stronger spatial dropout in CNN blocks
  - weight_decay: 1e-3 (vs 1e-4)  — stronger L2 penalty via AdamW
Expected outcome: lower training IoU, similar or better val IoU (generalisation).
Baseline (E0) mean IoU: 0.4968
"""

from dataclasses import replace

from configs.cnn_config import CNNConfig


def get_config():
    base = CNNConfig(experiment_name="e3_regularization")
    return replace(
        base,
        dropout=0.3,                # Changed: 0.1 → 0.3 (stronger spatial dropout)
        weight_decay=1e-3,          # Changed: 1e-4 → 1e-3 (stronger L2)
    )
