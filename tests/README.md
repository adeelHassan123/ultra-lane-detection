# `tests/` — Ensuring Structural Integrity

This folder contains the automated tests that prove the codebase is reliable and bug-free. We use `pytest` for all verification.

## 📂 Folder Structure

- **`test_models.py`**: Validates the neural architectures.
  - **Proof:** Ensures that a 256x256 image goes in and a 256x256 mask comes out without crashing.
- **`test_losses.py`**: Verifies that our custom loss functions (Combined, Tversky) output reasonable values (positive, non-zero).
- **`test_metrics.py`**: Tests the IoU and Dice score calculations against hand-calculated examples.
- **`test_dataset.py`**: Ensures the dataloaders are correctly resizing images and applying transforms.
- **`test_checkpoint.py`**: Proof of reliability. Verifies that we can save a model to disk and reload it with 100% identical weights.

## 🚀 How to Run
From the project root:
```bash
pytest
```
This will automatically find and run all `test_*.py` files.

## 🧠 Why Test?
In deep learning, "Silent Failures" are common (e.g., a loss function that stays constant).
- **Evidence of Quality:** By having a passing test suite, we provide **Proof** that the mathematical core of the project (Metrics/Losses) is correct.
- **Prevention:** Tests prevent "Regressions" (new code breaking old features).
