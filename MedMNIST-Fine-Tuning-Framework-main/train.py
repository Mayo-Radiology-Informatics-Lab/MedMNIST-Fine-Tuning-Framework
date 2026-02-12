#!/usr/bin/env python3
"""
MedMNIST Fine-Tuning Trainer (Improved Version)

Single-file PyTorch/timm trainer that supports:
- Backbones: resnet18, densenet121, vit_base_patch16_224, swin_tiny_patch4_window7_224
- Strategies: linear-probe (LP), full, partial (unfreeze last {20,30,40,50}%)
- Tasks: classification (single-label), multi-label (ChestMNIST), regression (BoneMNIST-REG, OrganAMNIST-REG)
- Metrics: Acc/AUC/F1 (classification), per-label AUC (multi-label), MAE/RMSE (regression)
- Schedulers: CosineAnnealingWarmRestarts, StepLR, ReduceLROnPlateau
- Logging: JSON + CSV + prints; saves best checkpoint by primary metric
- Early stopping support
- Improved error handling and validation

Expected folder layout (already built by your pipeline):
/scratch/MedMNist/
  <dataset>/
    training_info.json  # contains task_type, n_classes, split counts
    train/
      labels.pkl
      class_0/*.png ... or flat *.png
    val/
      labels.pkl
    test/
      labels.pkl

Usage (examples):
  python medmnist_trainer.py --dataset breastmnist --backbone resnet18 --strategy LP
  python medmnist_trainer.py --dataset chestmnist --backbone swin_tiny_patch4_window7_224 --strategy full
  python medmnist_trainer.py --dataset boneMNIST-reg --backbone vit_base_patch16_224 --strategy partial --unfreeze-pct 30
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
import random
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

try:
    import timm
except ImportError:
    logger.error("timm library not found. Please install with: pip install timm")
    sys.exit(1)


from src.experiment import init_experiment
from src.data import PNGFolderDataset, prepare_dataloader
from src.transforms import create_transforms
from src.utils import load_training_info, get_task_output_from_ds_name, plot_history_png
from src.model import init_model, log_model_analysis_structure
from src.optimizer import init_optimizer_scheduler
from src.callbacks import EarlyStopping, load_checkpoint, save_checkpoint
from src.metrics import compute_metrics, Metrics

# ---------------------
# Helpers & Configuration
# ---------------------
DATASET_ALIASES = {
    # classification
    'chestmnist': 'chestmnist',
    'breastmnist': 'breastmnist',
    'tissuemnist': 'tissuemnist',
    'organsmnist': 'organsmnist',
    'retinamnist': 'retinamnist',
    'pathmnist': 'pathmnist',
    'dermamnist': 'dermamnist',
    'octmnist': 'octmnist',
    'pneumoniamnist': 'pneumoniamnist',
    'organamnist': 'organamnist',
}

CLASS_MULTILABEL = {
    # MedMNIST v2 ChestMNIST has 14 labels (multi-label)
    'chestmnist': True,
}

REGRESSION_DATASETS = {
    'bonemnist-reg', 'organamnist-reg'
}

# Primary metric preference per task
PRIMARY_METRIC = {
    'classification': 'auc',
    'multilabel': 'auc_macro',  # macro of per-label AUCs
    'regression': 'mae',
}

# Backbone choices
BACKBONE_ALIASES = {
    'resnet18': 'resnet18',
    'resnet34': 'resnet34',
    'resnet50': 'resnet50',
    'densenet121': 'densenet121',
    'densenet161': 'densenet161',
    'efficientnet_b0': 'efficientnet_b0',
    'efficientnet_b4': 'efficientnet_b4',
    'vit_base_patch16_224': 'vit_base_patch16_224',
    'vit_small_patch16_224': 'vit_small_patch16_224',
    'swin_tiny_patch4_window7_224': 'swin_tiny_patch4_window7_224',
    'swin_small_patch4_window7_224': 'swin_small_patch4_window7_224',
}

IMG_SIZE = 224



# ---------------------
# Train / Eval loops
# ---------------------
def forward_logits(model: nn.Module, x: torch.Tensor, task: str) -> torch.Tensor:
    """Forward pass returning logits for the given task."""
    return model(x)

def build_criterion(task: str) -> nn.Module:
    """Build loss criterion for the given task."""
    if task == 'classification':
        return nn.CrossEntropyLoss()
    if task == 'multilabel':
        return nn.BCEWithLogitsLoss()
    if task == 'regression':
        return nn.MSELoss()  # Changed from L1Loss to MSELoss for better gradients
    raise ValueError(f"Unknown task: {task}")

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, 
                   scaler: Optional[torch.cuda.amp.GradScaler], criterion: nn.Module, 
                   device: torch.device, task: str) -> float:
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        
        if task == 'classification':
            # Ensure y is a 1D tensor of class indices
            if isinstance(y, (list, tuple)):
                y = torch.tensor(y, dtype=torch.long)
            elif isinstance(y, np.ndarray):
                y = torch.from_numpy(y).long()
            elif not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.long)
            else:
                y = y.long()
            
            # Flatten if needed but preserve batch dimension
            if y.dim() > 1:
                if y.size(-1) == 1:
                    y = y.squeeze(-1)
                else:
                    logger.warning(f"Unexpected label shape: {y.shape}, taking argmax")
                    y = y.argmax(dim=-1)
            
            y = y.to(device, non_blocking=True)
        else:
            y = y.to(device, dtype=torch.float32, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = forward_logits(model, x, task)
            loss = criterion(logits, y)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        num_batches += 1
    
    return running_loss / max(1, num_batches)

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, 
            task: str, criterion: nn.Module) -> Tuple[float, Metrics, np.ndarray, np.ndarray]:
    """Evaluate model on given dataset."""
    model.eval()
    all_logits = []
    all_targets = []
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            
            if task == 'classification':
                # Ensure y is a 1D tensor of class indices
                if isinstance(y, (list, tuple)):
                    y_tensor = torch.tensor(y, dtype=torch.long)
                elif isinstance(y, np.ndarray):
                    y_tensor = torch.from_numpy(y).long()
                elif not isinstance(y, torch.Tensor):
                    y_tensor = torch.tensor(y, dtype=torch.long)
                else:
                    y_tensor = y.long()
                
                # Flatten if needed but preserve batch dimension
                if y_tensor.dim() > 1:
                    if y_tensor.size(-1) == 1:
                        y_tensor = y_tensor.squeeze(-1)
                    else:
                        y_tensor = y_tensor.argmax(dim=-1)
                
                y_tensor = y_tensor.to(device, non_blocking=True)
                
                # Store original y for metrics (convert to numpy)
                if isinstance(y, torch.Tensor):
                    y_np = y.cpu().numpy()
                elif isinstance(y, np.ndarray):
                    y_np = y
                else:
                    y_np = np.array(y)
                
                # Ensure y_np is 1D for classification
                if y_np.ndim > 1:
                    if y_np.shape[-1] == 1:
                        y_np = y_np.squeeze(-1)
                    else:
                        y_np = y_np.argmax(axis=-1)
                        
            else:
                y_tensor = y.to(device, dtype=torch.float32, non_blocking=True)
                y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
            
            logits = forward_logits(model, x, task)
            loss = criterion(logits, y_tensor)
            
            running_loss += loss.item()
            num_batches += 1
            
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y_np)
    
    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    avg_loss = running_loss / max(1, num_batches)
    metrics = compute_metrics(task, targets, logits)
    
    return avg_loss, metrics, logits, targets



# ---------------------
# Main
# ---------------------
def main():
    parser = argparse.ArgumentParser(description="MedMNIST Fine-tuning Trainer")
    parser.add_argument('--base-dir', type=str, default='/scratch/MedMNist')
    parser.add_argument('--dataset', type=str, required=True, help='e.g., breastmnist, chestmnist, bonemnist-reg')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=list(BACKBONE_ALIASES.keys()))
    parser.add_argument('--strategy', type=str, default='LP', choices=['LP', 'full', 'partial'])
    parser.add_argument('--unfreeze-pct', type=int, default=20,choices=[10, 20, 30, 40, 50, 60],help='Percentage of blocks/layers to unfreeze when using --strategy partial')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-encoder', type=float, default=None, help='Optional different LR for encoder')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', type=str, default='/research/projects/Nico/FineTuningStrategies/01_Output')
    parser.add_argument('--amp', action='store_true', help='Enable mixed-precision training')
    parser.add_argument('--early-stopping', type=int, default=20, help='Early stopping patience (0 to disable)')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'plateau'])
    parser.add_argument('--direction', type=str, default='end',choices=['start', 'end'],help="Where to apply the partial unfreeze: 'end' (deeper layers) or 'start' (shallower layers)")
    parser.add_argument('--device' , type= int, default = 0)
    args = parser.parse_args()
    DEVICE = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{DEVICE}"





    # Setup Experiment
    output_dir, checkpoint_path, device = init_experiment(args)


    # Dataset setup
    ds_name = DATASET_ALIASES.get(args.dataset.lower(), args.dataset.lower())
    ds_dir = Path(args.base_dir) / ds_name
    
    if not ds_dir.exists():
        logger.error(f"Dataset folder not found: {ds_dir}")
        sys.exit(1)

    info = load_training_info(ds_dir)
    
    # Determine task and outputs
    task, n_outputs = get_task_output_from_ds_name(ds_name)
    logger.info(f"Task: {task}, Outputs: {n_outputs}, Dataset: {ds_name}")

    # Data transforms
    tf_train, tf_eval = create_transforms(IMG_SIZE)


    # Data loaders
    train_loader = prepare_dataloader(
        dataset_dir = ds_dir / 'train', 
        set_group = 'train',
        transforms = tf_train,
        task = task,
        batch_size = args.batch_size,
        num_workers = args.num_workers
    )

    val_loader = prepare_dataloader(
        dataset_dir = ds_dir / 'val', 
        set_group = 'val',
        transforms = tf_eval,
        task = task,
        batch_size = args.batch_size,
        num_workers = args.num_workers
    )
    test_loader = prepare_dataloader(
        dataset_dir = ds_dir / 'test', 
        set_group = 'test',
        transforms = tf_eval,
        task = task,
        batch_size = args.batch_size,
        num_workers = args.num_workers
    )


    # Model
    model_name = BACKBONE_ALIASES[args.backbone]
    model = init_model(
        model_name=model_name,
        n_outputs=n_outputs,
        strategy=args.strategy,
        unfreeze_pct=args.unfreeze_pct,
        direction=args.direction,
        device=device
    )
    # Analyze model structure after applying strategy
    total_params, trainable_params = log_model_analysis_structure(model)



    # Optimizer
    optimizer, scheduler = init_optimizer_scheduler(
        model = model,
        lr_classifier = args.lr,
        lr_encoder = args.lr_encoder,
        strategy = args.strategy,
        weight_decay = args.weight_decay,
        scheduler = args.scheduler,
        epochs = args.epochs
    )


    
    # Training setup
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = build_criterion(task)
    
    # Early stopping
    early_stopping = None
    if args.early_stopping > 0:
        early_stopping = EarlyStopping(patience=args.early_stopping, min_delta=1e-4)
        logger.info(f"Early stopping enabled with patience {args.early_stopping}")


    




    # Training loop
    best_primary = -np.inf
    best_epoch = 0
    history = []
    start_time = time.time()

    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, task)
        
        # Validate
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, device, task, criterion)
        
        # Schedule
        if args.scheduler == 'plateau':
            scheduler.step(val_metrics.primary)
        else:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        primary = val_metrics.primary
        

        # Check for improvement
        if primary > best_primary:
            best_primary = primary
            best_epoch = epoch

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_primary': float(best_primary),
                'val_loss': float(val_loss),
                'val_metrics': asdict(val_metrics),
                'args': vars(args),
                'task': task,
                'n_outputs': n_outputs,
            }

            try:
                save_checkpoint(checkpoint, checkpoint_path)
                logger.debug(f"✓ Overwrote best checkpoint at epoch {epoch} (primary={best_primary:.4f})")
            except Exception as e:
                logger.error(f"✗ Failed to save best checkpoint: {e}")

        # Logging
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'primary_metric': primary,
            'best_primary': best_primary,
            'epoch_time': epoch_time,
            'lr': optimizer.param_groups[0]['lr'],
        }
        
        # Add all validation metrics
        for key, value in asdict(val_metrics).items():
            if value is not None:
                log_entry[f'val_{key}'] = value
        
        history.append(log_entry)
        
        # Print progress
        metrics_str = f"primary={primary:.4f}"
        if val_metrics.acc is not None:
            metrics_str += f", acc={val_metrics.acc:.3f}"
        if val_metrics.auc is not None:
            metrics_str += f", auc={val_metrics.auc:.3f}"
        if val_metrics.mae is not None:
            metrics_str += f", mae={val_metrics.mae:.3f}"
        
        logger.info(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | {metrics_str} | best={best_primary:.4f} | {epoch_time:.1f}s")
        
        # Save history periodically
        if epoch % 5 == 0:
            pd.DataFrame(history).to_csv(output_dir / 'training_history.csv', index=False)
        
        # Early stopping check
        if early_stopping is not None:
            if early_stopping(primary):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break



    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.1f}s (best epoch: {best_epoch})")

    # Load best model for final evaluation
    model = load_checkpoint(model, checkpoint_path, device)


    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_loss, test_metrics, test_logits, test_targets = evaluate(model, test_loader, device, task, criterion)

    # Save final history
    pd.DataFrame(history).to_csv(output_dir / 'training_history.csv', index=False)
    plot_history_png(output_dir, history, task)

    # Prepare summary
    summary = {
        'dataset': ds_name,
        'backbone': args.backbone,
        'strategy': args.strategy,
        'unfreeze_pct': args.unfreeze_pct if args.strategy == 'partial' else 0,
        'Direction': args.direction if args.strategy == 'partial' else 0, 
        'task': task,
        'n_classes': n_outputs,
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs_trained': len(history),
        'best_epoch': int(best_epoch),
        'best_val_primary': float(best_primary),
        'test_loss': float(test_loss),
        'training_time_sec': float(total_time),
        'seed': args.seed,
    }

    # Add test metrics to summary
    for key, value in asdict(test_metrics).items():
        if value is not None:
            summary[f'test_{key}'] = float(value)

    # Save summary
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([summary]).to_csv(output_dir / 'summary.csv', index=False)

    # Save predictions and targets
    np.save(output_dir / 'test_predictions.npy', test_logits)
    np.save(output_dir / 'test_targets.npy', test_targets)

    # Save configuration
    config = {
        'args': vars(args),
        'model_config': {
            'name': model_name,
            'num_classes': n_outputs,
            'total_params': total_params,
            'trainable_params': trainable_params,
        },
        'dataset_info': info,
        'task_info': {
            'task_type': task,
            'primary_metric': PRIMARY_METRIC.get(task, 'unknown'),
        }
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Print final results
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Dataset: {ds_name}")
    logger.info(f"Model: {args.backbone}")
    strategy_line = args.strategy
    if args.strategy == 'partial':
        strategy_line += f" ({args.unfreeze_pct}%)"
    logger.info(f"Strategy: {strategy_line}")
    logger.info(f"Direction: {args.direction}")
    logger.info(f"Task: {task}")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Training time: {total_time:.1f}s")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info("-"*40)
    logger.info("Test Results:")
    if test_metrics.acc is not None:
        logger.info(f"  Accuracy: {test_metrics.acc:.4f}")
    if test_metrics.auc is not None:
        logger.info(f"  AUC: {test_metrics.auc:.4f}")
    if test_metrics.f1_macro is not None:
        logger.info(f"  F1-macro: {test_metrics.f1_macro:.4f}")
    if test_metrics.auc_macro is not None:
        logger.info(f"  AUC-macro: {test_metrics.auc_macro:.4f}")
    if test_metrics.mae is not None:
        logger.info(f"  MAE: {test_metrics.mae:.4f}")
    if test_metrics.rmse is not None:
        logger.info(f"  RMSE: {test_metrics.rmse:.4f}")
    logger.info(f"  Primary metric: {test_metrics.primary:.4f}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)

    return summary


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)