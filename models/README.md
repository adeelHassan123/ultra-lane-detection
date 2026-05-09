# `models/` — Neural Architectures

This folder contains the "blueprints" for the neural networks. We compare two main families: lightweight encoder-decoders and heavy-duty U-Nets.

## 📂 Folder Structure

- **`baseline_cnn.py`**: A custom, 4-block CNN. It lacks skip connections and has only ~200K parameters. It represents the "Floor" of performance (E0-E5).
- **`unet_scratch.py`**: A classic U-Net implementation with skip connections, built from scratch. It has ~31M parameters (E6).
- **`unet_smp.py`**: A professional U-Net using the `segmentation-models-pytorch` library. It allows us to use **Pretrained Encoders** (like ResNet-34) for Transfer Learning (E7-E9).
- **`activations.py`**: A factory for switching between `ReLU`, `LeakyReLU`, and `GELU`.
- **`factory.py`**: The `build_model()` function that instantiates the architecture based on the config.

## 🧠 Architectural Insights

### 1. The Power of Skip Connections
- **The Problem:** In deep CNNs, spatial details (like the exact location of a thin lane) are lost during downsampling (pooling).
- **The Proof:** `UNetScratch` outperforms `BaselineCNN` because its "Skip Connections" pass high-resolution details from the encoder directly to the decoder.

### 2. Transfer Learning (The E7 Breakthrough)
- **The Strategy:** Instead of training a model from zero, we start with a model that already knows how to "see" (pretrained on ImageNet).
- **The Evidence:** E7 (Transfer Learning) typically achieves 10-15% higher IoU than E6 (Scratch) because it starts with a feature-rich "brain."

### 3. Efficiency vs. Performance
| Model | Params | Accuracy (IoU) | Best For |
| :--- | :--- | :--- | :--- |
| **Baseline CNN** | ~0.2M | Lower | Edge Devices / Real-time |
| **U-Net Scratch** | ~31M | High | High-accuracy research |
| **U-Net + ResNet34**| ~24M | **Highest** | **Production Grade** |

## 🚀 Key Methods
In `unet_smp.py`, we expose:
- `get_encoder_params()`
- `get_decoder_params()`

This allows the `Trainer` to use **Differential Learning Rates** (e.g., training the encoder slowly and the decoder quickly), which is a "Proof-backed" best practice for Transfer Learning.
