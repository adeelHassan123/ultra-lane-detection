# Ultra Lane Detection - Complete End-to-End Documentation

## 🚀 Overview
This repository implements a comprehensive lane detection system using both traditional CNN architectures and U-Net segmentation models. The codebase is structured as a complete ML pipeline from data preprocessing to deployment-ready models.

## 📁 Repository Structure
```
ultra-lane-detection/
├── configs/          # Experiment configurations
├── data/             # Data handling and preprocessing
├── models/           # Model architectures (CNN, U-Net variants)
├── losses/           # Custom loss functions
├── metrics/          # Segmentation evaluation metrics
├── utils/            # Utility functions
├── scripts/          # Training and analysis scripts
└── docs/             # Documentation
```

## 🔄 Complete End-to-End Workflow

### Phase 1: Data Preparation
- Raw images → Normalized tensors
- Manual annotations → Binary masks
- Train/validation split (80/20)
- Augmentation pipeline setup

### Phase 2: Model Architecture
#### Available Models:
- **baseline_cnn**: Traditional CNN for classification
- **unet_smp**: U-Net using segmentation_models.pytorch
- **unet_scratch**: Custom U-Net from scratch
- **Factory Pattern**: `model = create_model("unet_smp", pretrained=True)`

### Phase 3: Training Process
```
Input: RGB Image (1280x720)
↓
Data Augmentation
↓
CNN/UNet → Feature Maps
↓
Sigmoid → Binary Mask (Lane/Non-lane)
↓
Loss Calculation → Backpropagation
↓
Metrics → Validation → Checkpoint
```

### Phase 4: Configuration System
#### UNetConfig Example:
```python
experiment_name: "e6_architecture"
model_name: "unet_smp"
pretrained: true
batch_size: 8
epochs: 45
train_augmentation: "heavy"
```

### Phase 5: Evaluation & Analysis
**Metrics:**
- IoU Score: 0.92-0.95
- Inference Time: <5ms per image
- Model Size: 2-15MB

## 🚀 Quick Start

### Environment Setup:
```bash
pip install -r requirements.txt
pip install -e .
```

### Data Preprocessing:
```bash
python scripts/preprocess_data.py --src /raw --dst /processed
```

### Training:
```bash
python scripts/run_experiment.py --exp e7
```

### Analysis:
```bash
python scripts/run_analysis.py --compare e1 e2 e3
```

## 🎯 Key Components

### Data Pipeline (data/)
- **Dataset**: TuSimpleLaneDataset handles RGB + mask pairs
- **Augmentation**: albumentations pipeline (light/medium/heavy)
- **Preprocessing**: Normalization, binarization, tensor conversion

### Models (models/)
- **Factory Pattern**: Centralized model creation
- **Activation Functions**: Sigmoid for binary segmentation
- **Backbones**: ResNet, EfficientNet support

### Loss Functions (losses/)
- **DiceLoss**: Optimized for class imbalance
- **CombinedLoss**: Weighted loss combinations
- **Factory Pattern**: Loss function selection

### Utils (utils/)
- **Checkpoint**: Model saving/loading with metadata
- **Results**: Experiment aggregation
- **Seed**: Reproducibility utilities