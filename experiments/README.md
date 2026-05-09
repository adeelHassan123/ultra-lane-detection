# `experiments/` — The Scientific Lab

This folder defines the specific research studies (E0 through E9). It is the core of the "Research" aspect of this project, where each file represents a scientific hypothesis.

## 📂 Folder Structure

- **`e0_baseline.py`**: The control group. A simple CNN with minimal features.
- **`e1_optimizer.py`**: Testing Adam vs. AdamW.
- **`e2_activation.py`**: Testing ReLU vs. GELU.
- **`e3_regularization.py`**: Testing Dropout and Weight Decay.
- **`e4_augmentation.py`**: Testing the impact of "Heavy" data augmentation.
- **`e5_batchnorm.py`**: Testing the stability of BatchNorm.
- **`e6_architecture.py`**: The jump to **U-Net**. Testing skip-connections from scratch.
- **`e7_transfer.py`**: **The Best Model.** Testing Transfer Learning with a pretrained ResNet-34 backbone.
- **`e8_loss.py`**: Testing specialized DiceBCE loss.
- **`e9_ablation.py`**: The "Ablation Study". Removing augmentation from E7 to see its isolated impact.

## 🧠 The Scientific Method in Code

Each experiment file follows a strict pattern:
1. **Hypothesis:** Stated in the docstring at the top.
2. **Variable:** The single change made (e.g., `optimizer_name="adam"`).
3. **Execution:** Handled by the `run_experiment()` function in `__init__.py`.

## 🚀 How to Run
Use the unified runner script:
```bash
python scripts/run_experiment.py --exp e7
```
This will:
- Load the config for E7.
- Train the model across **3 different seeds** (42, 123, 777).
- Log the average results to `outputs/results/master_results.csv`.

## 📊 Evidence & Proof
By running each experiment for 3 different seeds, we provide **Proof** that the results aren't just "lucky". The `std_iou` (standard deviation) tells us how stable the model is.
