# `metrics/` — Quantifying Success

This folder defines the "Yardsticks" used to measure how well our models are performing. We don't just look at "Accuracy"; we look at specialized metrics for segmentation and efficiency.

## 📂 Folder Structure

- **`segmentation.py`**: The core metrics.
  - **IoU (Intersection over Union)**: The "Gold Standard" for lane detection.
  - **Dice Score**: Measures overlap (closely related to IoU).
  - **Pixel Accuracy**: Percentage of correctly classified pixels.
- **`advanced.py`**: 
  - **AUC-IoU**: Calculates IoU across multiple thresholds (0.3 to 0.7) to ensure the model is robust and not just lucky at one specific threshold.
  - **Precision/Recall/F1**: Standard classification metrics.
- **`efficiency.py`**:
  - **Parameter Counting**: Proof of model size (Complexity).
  - **Inference Time**: Proof of real-world speed (Latency).

## 🧠 Metric Definitions & Proof

### 1. Intersection over Union (IoU)
- **Calculation:** $\frac{Area\_of\_Overlap}{Area\_of\_Union}$
- **Why?** Accuracy is misleading in lane detection. If the model predicts "all black", accuracy is 99%. IoU would be 0%. 
- **Evidence:** We use IoU as our primary metric because it strictly penalizes missing the lane and predicting "ghost" lanes.

### 2. AUC-IoU (Area Under Curve)
- **Proof of Robustness:** By averaging IoU at thresholds of `0.3, 0.4, 0.5, 0.6, 0.7`, we prove that our model's predictions have high confidence and clean edges.

### 3. Inference Time
- **Real-world Proof:** We measure how many milliseconds it takes for the model to process one image on the CPU/GPU. A model that is 90% accurate but takes 1 second to run is useless for a self-driving car.

## 🚀 Usage in Trainer
The `Trainer` class in `training/trainer.py` calls `compute_metrics()` at the end of every epoch and logs them to the CSV file.
