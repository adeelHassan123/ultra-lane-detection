# `scripts/` Guide

CLI entrypoints for the project.

## Files

- `preprocess_data.py`
  - one-time data prep: copy + mask dilation + validation
- `run_experiment.py`
  - launches a selected experiment (`--exp e0 ... e9`)
- `run_analysis.py`
  - generates post-training figures and summary files

## Principle

These scripts should remain thin wrappers over modular library code from other folders.
