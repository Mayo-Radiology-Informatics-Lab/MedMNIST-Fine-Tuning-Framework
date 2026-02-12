# MedMNIST Fine-Tuning Framework

A PyTorch-based framework for fine-tuning pretrained models on MedMNIST datasets with configurable training strategies.

## Features

- **Multiple Backbones**: ResNet, DenseNet, EfficientNet, ViT, Swin Transformer
- **Training Strategies**: Linear Probing (LP), Full Fine-tuning, Partial Fine-tuning
- **Task Support**: Single-label classification, Multi-label classification, Regression
- **Automatic Metrics**: AUC, Accuracy, F1-score, MAE, RMSE (task-dependent)
- **Training Utilities**: Mixed-precision training, early stopping, learning rate scheduling

## Requirements

```bash
pip install torch torchvision timm numpy pandas scikit-learn matplotlib pillow
```

## Dataset Structure

```
<base_dir>/
└── <dataset_name>/
    ├── training_info.json    # Optional: contains n_classes, task_type
    ├── train/
    │   ├── labels.pkl        # Pickled label array
    │   └── *.png             # Images (flat or in class_* subdirs)
    ├── val/
    │   ├── labels.pkl
    │   └── *.png
    └── test/
        ├── labels.pkl
        └── *.png
```

## Quick Start

```bash
# Linear Probing on BreastMNIST
python train.py --dataset breastmnist --backbone resnet18 --strategy LP

# Full Fine-tuning on PathMNIST
python train.py --dataset pathmnist --backbone densenet121 --strategy full

# Partial Fine-tuning (30% layers) on DermaMNIST
python train.py --dataset dermamnist --backbone vit_base_patch16_224 --strategy partial --unfreeze-pct 30
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | *required* | Dataset name (e.g., `breastmnist`, `chestmnist`) |
| `--backbone` | str | `resnet18` | Model architecture (see [Backbones](#supported-backbones)) |
| `--strategy` | str | `LP` | Training strategy: `LP`, `full`, `partial` |
| `--unfreeze-pct` | int | `20` | Layers to unfreeze for `partial` strategy: 10, 20, 30, 40, 50, 60 |
| `--direction` | str | `end` | Unfreeze direction for `partial`: `start` or `end` |
| `--base-dir` | str | `/scratch/MedMNist` | Root directory containing datasets |
| `--out-dir` | str | `./output` | Output directory for results |
| `--batch-size` | int | `64` | Training batch size |
| `--epochs` | int | `500` | Maximum training epochs |
| `--lr` | float | `1e-4` | Learning rate (classifier head) |
| `--lr-encoder` | float | `None` | Separate learning rate for encoder (optional) |
| `--weight-decay` | float | `1e-4` | Weight decay for AdamW optimizer |
| `--scheduler` | str | `cosine` | LR scheduler: `cosine`, `step`, `plateau` |
| `--early-stopping` | int | `20` | Early stopping patience (0 to disable) |
| `--amp` | flag | `False` | Enable mixed-precision training |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--num-workers` | int | `8` | DataLoader workers |
| `--device` | int | `0` | CUDA device index |

## Training Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **LP** (Linear Probing) | Freeze encoder, train only classifier head | Limited data, fast training |
| **full** | Train all parameters | Large datasets, maximum performance |
| **partial** | Unfreeze top-k% of encoder blocks + classifier | Balance between LP and full |

### Partial Fine-tuning Direction

- `--direction end`: Unfreeze deeper layers (closer to output) — *recommended*
- `--direction start`: Unfreeze shallower layers (closer to input)

## Supported Backbones

| Architecture | Model Names |
|--------------|-------------|
| ResNet | `resnet18`, `resnet34`, `resnet50` |
| DenseNet | `densenet121`, `densenet161` |
| EfficientNet | `efficientnet_b0`, `efficientnet_b4` |
| Vision Transformer | `vit_base_patch16_224`, `vit_small_patch16_224` |
| Swin Transformer | `swin_tiny_patch4_window7_224`, `swin_small_patch4_window7_224` |

## Supported Datasets

| Dataset | Task | Classes/Outputs |
|---------|------|-----------------|
| `breastmnist` | Classification | 2 |
| `pneumoniamnist` | Classification | 2 |
| `retinamnist` | Classification | 5 |
| `octmnist` | Classification | 4 |
| `dermamnist` | Classification | 7 |
| `pathmnist` | Classification | 9 |
| `tissuemnist` | Classification | 8 |
| `organamnist` | Classification | 11 |
| `organsmnist` | Classification | 11 |
| `chestmnist` | Multi-label | 14 |

## Output Structure

```
<out_dir>/<run_name>/
├── checkpoints/
│   └── best.pt              # Best model checkpoint
├── training_history.csv     # Per-epoch metrics
├── summary.json             # Final results summary
├── summary.csv              # Summary in CSV format
├── config.json              # Full configuration
├── test_predictions.npy     # Test set predictions
├── test_targets.npy         # Test set ground truth
├── loss_curves.png          # Training/validation loss plot
└── val_primary.png          # Primary metric plot
```

## Example Experiments

```bash
# Experiment 1: Compare strategies on BreastMNIST
for strategy in LP full partial; do
    python train.py --dataset breastmnist --backbone resnet18 --strategy $strategy --seed 42
done

# Experiment 2: Partial fine-tuning sweep
for pct in 10 20 30 40 50; do
    python train.py --dataset dermamnist --backbone vit_base_patch16_224 \
        --strategy partial --unfreeze-pct $pct --seed 42
done

# Experiment 3: Multi-seed evaluation
for seed in 42 123 456; do
    python train.py --dataset pathmnist --backbone densenet121 \
        --strategy full --seed $seed
done
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2025finetuning,
  title={Your Paper Title},
  author={Nicolo Pecco and Collaborators},
  journal={Journal/Conference},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

