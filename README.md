# DINOv2-Based Semantic Segmentation

    This project implements a semantic segmentation pipeline using [DINOv2](https://github.com/facebookresearch/dinov2) as a feature extractor, with multiple decoder heads including U-Net-style, DeepLabv3+, and attention-based decoders. It supports custom loss functions (Dice, Cross Entropy, Lovász) and provides visualization tools.

## Project Structure
    ProjectBytes/
    ├── src/
    │ ├── model.py # DINOv2 + decoder heads
    │ ├── train.py # Training loop and evaluation
    │ ├── inference.py # Inference on custom images
    │ ├── attention_viz.py # DINOv2 attention visualization
    │ ├── diceCE.py # Dice + CrossEntropy Loss
    │ ├── lovasz_losses.py # Lovász Loss
    │ ├── dataset.py # VOC dataset loader
    │ └── val_preds/ # Saved predictions from validation set
    ├── airplane1.jpg # Sample test image
    ├── dino_seg.ipynb # Experiment notebook
    └── README.md

## Setup Instructions

    1. Install required packages: pip install -r requirements.txt
    2. Change dir to train.py: cd src/
    3. Train the model: python train.py
    4. Run Inference: python inference.py

## Model Design

    Backbone: DINOv2-Large (ViT-based)
    Heads:
        Custom UNet-style decoder
        DeepLabv3+ decoder for benchmarking
        Attention-based decoder
    Loss:
        Dice Loss
        Dice + CrossEntropy Loss
        Lovász Softmax Loss
    Evaluation: mean IoU, pixel accuracy

## Visualizations

    Attention maps from DINOv2 are saved during inference
    Validation predictions are saved in val_preds/
    Training/Validation loss plotted as loss_plot.png




