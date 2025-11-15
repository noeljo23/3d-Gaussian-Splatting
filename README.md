# Gaussian Splats VAE: 3D Object Generation from Multi-View Images

A Variational Autoencoder that learns to generate 3D objects represented as Gaussian splats from multi-view 2D images.

## Quick Start

### Dependencies
```bash
pip install torch torchvision tqdm pillow numpy
```

### Training
```bash
python train.py
```

### Evaluation
```bash
python evaluate.py
```

## Project Structure
- `train.py` - Full training pipeline (1000 objects, 100 epochs)
- `evaluate.py` - Evaluation metrics and analysis
- `quick_train.py` - Quick validation on 50 objects
- `main.py` - VAE architecture classes

## Dataset
Download ShapeNet Core v2 from https://shapenet.org/
Category ID for chairs: `03001627`

Render multi-view images using multiview-renderer:
https://github.com/vencia/multiview-renderer

## Results
- PSNR: 13.20 dB
- SSIM: 0.9544
- Latent dimension: 64
- Number of splats: 2048 per object
