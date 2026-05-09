# `configs/` — Centralized Experiment Control

This folder is the "Brain" of the project. It uses Python `dataclasses` to ensure that every experiment is reproducible, type-safe, and easy to modify without touching the core logic.

## 📂 Folder Structure

- **`base.py`**: The `BaseConfig` class. It contains all "Global Defaults" like paths, image size (256x256), training seeds (42, 123, 777), and learning rates.
- **`cnn_config.py`**: Inherits from `BaseConfig`. Specifically tuned for the lightweight CNN baseline (e.g., higher batch size of 16).
- **`unet_config.py`**: Inherits from `BaseConfig`. Optimized for the heavy U-Net models (e.g., lower batch size of 8, more epochs).
- **`augmentation_config.py`**: Defines the specific operations for `light` (FLIP + BRIGHTNESS) and `heavy` (FLIP + BRIGHTNESS + BLUR + NOISE) policies.

## 🧠 Why this approach?

### 1. Evidence of Reproducibility
Every experiment (E0–E9) corresponds to a specific configuration. By saving these configs with the model, we provide "Proof" that anyone can recreate the results by simply loading the same config.

### 2. Surgical Modifications
Instead of passing 50 arguments to a function, we pass one `cfg` object. This makes the code cleaner and less prone to "silent bugs" where a parameter is forgotten.

### 3. Dynamic Path Resolution
`BaseConfig` automatically creates the necessary output folders (`logs`, `checkpoints`, `figures`) using the `ensure_output_dirs()` method. This ensures the project runs smoothly on any machine (Local or Google Colab).

## 🛠️ Key Parameters Explained

| Parameter | Default | Why it matters |
| :--- | :--- | :--- |
| `seed` | `42` | Ensures "Randomness" is the same every time. |
| `amp` | `True` | **Automatic Mixed Precision**. Speeds up training on modern GPUs by 2x. |
| `data_root` | `/content/fast_data` | Points to the ultra-fast NumPy dataset for 10x I/O speed. |
| `tversky_alpha` | `0.3` | Controls how much we penalize False Positives (over-segmentation). |

## 🚀 How to use
To create a new experiment, you don't edit these files. Instead, you use `replace()` in the experiment scripts:
```python
new_cfg = replace(base_cfg, learning_rate=1e-4, experiment_name="my_test")
```
This keeps the original configs "Pure" and unmodified.