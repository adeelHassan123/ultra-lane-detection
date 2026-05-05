"""
E1: Optimizer Comparison
========================
Variable changed:  optimizer_name  → "adam"  (vs default "adamw" in E0)
Hypothesis: Adam without weight-decay decoupling may converge faster on this
            small dataset but plateau earlier due to adaptive LR drift.
Expected outcome: slight IoU drop vs E0, establishing AdamW as the better choice.
Baseline (E0) mean IoU: 0.4968
"""

from dataclasses import replace

from configs.cnn_config import CNNConfig


def get_config():
    base = CNNConfig(experiment_name="e1_optimizer")
    return replace(
        base,
        optimizer_name="adam",      # Changed: Adam vs AdamW (E0 default)
        learning_rate=1e-3,         # Same as E0
        weight_decay=1e-4,          # Same as E0 (ignored by Adam, but kept for parity)
    )
