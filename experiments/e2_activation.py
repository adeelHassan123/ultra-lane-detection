"""
E2: Activation Function Comparison
====================================
Variable changed:  activation  → "gelu"  (vs default "relu" in E0)
Hypothesis: GELU's smooth gradient flow (used in transformers/modern CNNs) may
            help with thin lane structures that ReLU's hard zero can miss.
Expected outcome: marginal improvement or parity with E0; diminishing returns
                  for activation choice vs architecture choice.
Baseline (E0) mean IoU: 0.4968
"""

from dataclasses import replace

from configs.cnn_config import CNNConfig


def get_config():
    base = CNNConfig(experiment_name="e2_activation")
    return replace(
        base,
        activation="gelu",          # Changed: GELU vs ReLU (E0 default)
    )
