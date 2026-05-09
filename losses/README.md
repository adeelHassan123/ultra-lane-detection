# `losses/` — Objective Functions & Optimization

This folder contains the custom loss functions that guide the model's learning. Lane detection is a highly imbalanced task (mostly background, very few lane pixels), and these losses are designed to solve that.

## 📂 Folder Structure

- **`dice_bce.py`**: A hybrid loss. Combines pixel-wise Binary Cross Entropy (BCE) with the Dice coefficient (which focuses on the overlap area).
- **`tversky.py`**: An advanced loss that allows us to control the trade-off between False Positives (over-detection) and False Negatives (missing lanes).
- **`combined.py`**: **The Best Loss.** It uses a **Dynamic Positive Weight** for BCE based on the ratio of background-to-lane pixels in each batch, combined with Tversky loss.
- **`factory.py`**: The `build_loss()` function that picks the right loss based on the `cfg.loss_name`.

## 🧠 The Math of "Evidence"

### 1. Imbalance Handling
In a 256x256 image, there might be 65,536 pixels, but only 500 are "Lane". 
- Standard BCE would ignore the lane and get 99% accuracy by predicting "Background" everywhere.
- **Proof of Solution:** Our `CombinedLoss` dynamically calculates a `pos_weight` (e.g., 100.0) to tell the model: "Missing a lane pixel is 100x worse than missing a background pixel."

### 2. Tversky vs. Dice
- **Dice Loss** treats False Positives and False Negatives equally.
- **Tversky Loss** (with $\alpha=0.3, \beta=0.7$) penalizes False Negatives (missing lanes) more heavily. This is "Proof" of our safety-first approach: it's better to detect a slightly noisy lane than to miss one entirely.

## 🚀 How to Swap
Simply change `loss_name` in your config:
```python
cfg.loss_name = "tversky"
```
The `factory.py` will handle the instantiation automatically.
