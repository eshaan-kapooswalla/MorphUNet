# MorphUNet

A comprehensive PyTorch implementation of various U-Net architectures for image segmentation tasks, with a focus on the Carvana Image Masking Challenge dataset.

## Overview

MorphUNet is a deep learning framework that implements multiple U-Net variants for semantic segmentation tasks. The project includes implementations of:

- **U-Net**: Classic U-Net architecture
- **U-Net 2+**: Enhanced U-Net with additional skip connections
- **U-Net 3+**: Advanced U-Net with full-scale skip connections and deep supervision
- **U-Net 3+ Deep Supervision**: U-Net 3+ with deep supervision capabilities
- **U-Net 3+ Deep Supervision with CGM**: U-Net 3+ with deep supervision and class-guided module

## Features

- **Multiple U-Net Architectures**: Support for various U-Net variants
- **Flexible Data Loading**: Custom dataset loader for image-mask pairs
- **Multiple Loss Functions**: BCE, IoU, and MS-SSIM loss implementations
- **Real-time Visualization**: Training progress visualization with matplotlib
- **GPU Support**: Automatic CUDA detection and utilization
- **Modular Design**: Clean separation of models, loss functions, and utilities

## Project Structure

```
MorphUNet/
├── data_loader/
│   └── data_loader.py          # Custom dataset loader for Carvana challenge
├── models/
│   ├── UNET.py                 # Classic U-Net implementation
│   ├── UNet_2Plus.py           # U-Net 2+ architecture
│   ├── UNet_3Plus.py           # U-Net 3+ architecture
│   ├── UNet_3Plus_deep_supervision.py
│   ├── UNet_3Plus_deep_sup_cgm.py
│   ├── layers.py               # Common U-Net layers
│   └── init_weights.py         # Weight initialization utilities
├── down_samplers/
│   └── UNET_MaxPool.py         # MaxPool downsampling
├── up_samplers/
│   ├── UNET_Bilinear.py        # Bilinear upsampling
│   └── UNET_Transpose.py       # Transpose convolution upsampling
├── loss/
│   ├── bceLoss.py              # Binary Cross Entropy loss
│   ├── iouLoss.py              # Intersection over Union loss
│   └── msssimLoss.py           # MS-SSIM loss
├── display_helper.py           # Training visualization utilities
├── main.py                     # Main training script
├── setup.py                    # Configuration parameters
└── README.md                   # This file
```

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- PIL (Pillow)
- matplotlib
- tqdm

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MorphUNet
```

2. Install dependencies:
```bash
pip install torch torchvision numpy pillow matplotlib tqdm
```

## Data Setup

The project is configured for the Carvana Image Masking Challenge dataset. Place your data in the following structure:

```
carvana-image-masking-challenge/
├── train/
│   └── train/                  # Training images
└── train_masks/
    └── train_masks/            # Training masks
```

## Usage

### Training

Run the main training script:

```bash
python main.py
```

### Configuration

Modify `setup.py` to adjust training parameters:

```python
learning_rate = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 5
num_epochs = 5
num_workers = 1
image_scaling = 8
```

### Model Selection

In `main.py`, you can choose between different U-Net architectures:

```python
# Classic U-Net
model = UNET.UNET(features, num_inp_channels, num_labels).to(device)

# U-Net 3+ (default)
model = models.UNet_3Plus.UNet_3Plus().to(device)
```

## Architecture Details

### U-Net 3+ Features

- **Full-scale Skip Connections**: Connects encoder and decoder at all scales
- **Deep Supervision**: Multiple output paths for better gradient flow
- **Class-guided Module**: Enhanced feature learning for better segmentation
- **Flexible Feature Scaling**: Configurable feature dimensions

### Loss Functions

- **BCE Loss**: Binary Cross Entropy for binary segmentation
- **IoU Loss**: Intersection over Union loss for better boundary accuracy
- **MS-SSIM Loss**: Multi-scale structural similarity loss

## Training Visualization

The training process includes real-time visualization showing:
- Target images
- Target masks
- Predicted outputs
- Loss curves

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request






## Acknowledgments

- Original U-Net paper: Ronneberger et al. (2015)
- U-Net 3+ paper: Huang et al. (2020)
- Carvana Image Masking Challenge dataset
