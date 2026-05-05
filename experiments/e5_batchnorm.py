"""
E5: Batch Normalisation Study
==============================
Variable changed:  use_batchnorm  → False  (vs True in E0)
Hypothesis: Removing BatchNorm tests how much the model depends on normalised
            activations. Lane detection's small lane-to-background ratio makes
            BN critical for stable gradient flow — removing it should hurt.
Expected outcome: lower IoU and slower/unstable convergence, proving BN's value.
Baseline (E0) mean IoU: 0.4968
"""

from dataclasses import replace

from configs.cnn_config import CNNConfig


def get_config():
    base = CNNConfig(experiment_name="e5_batchnorm")
    return replace(
        base,
        use_batchnorm=False,        # Changed: True → False (ablate BatchNorm)
    )
