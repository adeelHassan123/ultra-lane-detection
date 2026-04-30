from dataclasses import replace

import numpy as np

from training import Trainer


def run_experiment(cfg, seeds=None):
    if seeds is None:
        seeds = cfg.seeds
    best_scores = []
    for seed in seeds:
        run_cfg = replace(cfg, seed=seed)
        trainer = Trainer(run_cfg)
        score = trainer.fit()
        best_scores.append(score)
    return {
        "mean_iou": float(np.mean(best_scores)),
        "std_iou": float(np.std(best_scores)),
        "scores": [float(s) for s in best_scores],
    }
