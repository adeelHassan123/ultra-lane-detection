# `data/` — The Optimized Data Pipeline

This folder manages the entire lifecycle of the dataset: from raw ingestion and dilation to high-speed loading during training.

## 📂 Folder Structure

- **`dataset.py`**: The `TuSimpleLaneDataset`. A standard PyTorch dataset that reads `.jpg` images and `.png` masks.
- **`fast_dataset.py`**: **The Speed Secret.** A specialized dataset that reads preprocessed `.npz` (NumPy) files. This is **10-20x faster** than reading JPEGs, ensuring the GPU never waits for data.
- **`transforms.py`**: Defines the "Wait, how did it learn that?" logic. Uses the `Albumentations` library for heavy-duty augmentations (Motion Blur, Gaussian Noise, etc.).
- **`dataloader.py`**: Standard dataloader factory.
- **`preprocess.py`**: Contains `dilate_masks()`. This is **crucial evidence** for the project: lane markings are often too thin (1-2 pixels) for a CNN to learn. We dilate them to 15 pixels to provide a stronger signal.
- **`validate.py`**: A "Sanity Guard". It checks every image/mask pair for corruption, shape mismatch, or "empty" masks before training starts.

## 🚀 Key Workflows

### 1. Preprocessing (One-time)
Run `python scripts/preprocess_data.py`. This:
- Copies images to a local folder.
- **Dilates** masks to make lanes thicker.
- Runs **Validation** to ensure 100% data integrity.

### 2. NumPy Optimization
Run `python scripts/preprocess_to_numpy.py`. This converts the entire dataset into binary NumPy format.
- **Proof of Speed:** Reduces epoch time from ~25 minutes (on Google Drive) to ~3 minutes.

## 🛠️ Data Augmentation Policies

| Policy | Operations | Purpose |
| :--- | :--- | :--- |
| **None** | Resize, Normalize | Standard validation (no noise). |
| **Light** | Flip, Brightness | Basic variety for small models. |
| **Heavy** | Blur, Noise, Contrast | Forces the model to learn "Shape" rather than "Color". Used in E4 and E7. |

## 🧠 Why "Dilation"?
In lane detection, the "Signal-to-Noise" ratio is very low (lanes take up <1% of pixels). 
- **Evidence:** Without dilation, Loss functions (like BCE) often ignore the lanes entirely. 
- **Proof:** By dilating masks, we increase the "Lane Pixel" count, helping the model converge faster.

