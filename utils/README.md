# `utils/` — Cross-Cutting Utilities

This folder contains helper functions and classes that are used across the entire project. They handle low-level tasks like logging, seed management, and hardware detection.

## 📂 Folder Structure

- **`logger.py`**: Contains `CSVLogger` and `ResultsLogger`. These are the "Truth Keepers" that write training metrics to disk, providing the **Proof** for all our charts.
- **`device.py`**: Automatically detects if you are on `CUDA` (NVIDIA GPU), `MPS` (Apple Silicon), or `CPU`. Ensures the code is "Hardware Agnostic."
- **`seed.py`**: The "Repeatability Guard". It sets the random seed for Python, NumPy, and PyTorch. 
  - **Evidence:** Crucial for research. If we run the same experiment twice, we get identical results because the seed is fixed.
- **`checkpoint.py`**: Low-level helper for saving/loading PyTorch `state_dicts`.
- **`results.py`**: Helpers for parsing and saving CSV result files.

## 🧠 Key Utilities & Proof

### 1. The Power of `seed.py`
In research, it's vital to know if a 2% improvement is due to your code or just "lucky randomness". 
- **Proof:** By fixing the seed to `42, 123, 777`, we ensure that every model starts with the same initial weights and sees the same data order. This makes our comparisons "Fair."

### 2. `CSVLogger` for Evidence
- **Proof of Performance:** Every single epoch's loss and IoU is recorded. This raw data is what allows us to generate the "Training Curves" that prove the model is actually learning.

### 3. Hardware Auto-Detection
- **Evidence of Flexibility:** The `get_device()` function in `device.py` means this project can run on a high-end server or a developer's laptop without changing a single line of code.

## 🚀 Usage
These are typically imported as:
```python
from utils.device import get_device
from utils.seed import set_all_seeds

device = get_device()
set_all_seeds(42)
```
