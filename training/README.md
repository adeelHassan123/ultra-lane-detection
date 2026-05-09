# `training/` — The Engine Room

This folder contains the core logic for the training loop. It is where data, models, and losses come together to perform the "Learning" process.

## 📂 Folder Structure

- **`trainer.py`**: The "Orchestrator". The `Trainer` class manages everything: device placement (GPU), optimizer setup, differential learning rates, and the main training loop.
- **`train_epoch.py`**: Contains the logic for a single pass through the training data. Includes **Automatic Mixed Precision (AMP)** for speed.
- **`validate_epoch.py`**: Logic for evaluating the model on unseen data.
- **`early_stopping.py`**: Proof of Efficiency. It monitors the validation IoU and stops training if the model stops improving, saving hours of GPU time.
- **`checkpoint.py`**: Manages saving and loading the best model weights.

## 🧠 Advanced Training Features

### 1. Differential Learning Rates
- **Strategy:** In `trainer.py`, we can train the **Encoder** (backbone) with a lower learning rate (e.g., 1e-4) and the **Decoder** with a higher one (e.g., 1e-3).
- **Proof:** This prevents "forgetting" the valuable pretrained ImageNet features while allowing the decoder to learn the new task of lane detection quickly.

### 2. Automatic Mixed Precision (AMP)
- **Proof of Speed:** We use `torch.amp` to perform calculations in `float16` instead of `float32`. This reduces GPU memory usage and doubles training speed on modern GPUs.

### 3. Torch Compile
- **The Optimization:** If PyTorch 2.0+ is available, we use `torch.compile(model)`.
- **Evidence:** This provides a **20-30% speedup** by optimizing the model's graph specifically for the hardware.

## 🚀 The Life of an Epoch
1. **`train_one_epoch`**: Calculate loss → backprop → update weights.
2. **`validate`**: Check performance on validation set.
3. **`checkpoint`**: If Val IoU is the best so far, save it!
4. **`early_stopping`**: If no improvement for 10 epochs, stop.
