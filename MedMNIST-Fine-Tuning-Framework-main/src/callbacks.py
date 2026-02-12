import os
import logging
import torch
from torch import nn
from pathlib import Path
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

__all__ = [
    'EarlyStopping',
    'save_checkpoint',
    'load_checkpoint',
]



class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            
        self.early_stop = self.counter >= self.patience
        return self.early_stop



def save_checkpoint(checkpoint: dict, path: Path):
    """Write to a temp file then atomically replace target to avoid partial writes."""
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)  # atomic on POSIX and modern Windows
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass




def load_checkpoint(model, checkpoint_path: Path, device) -> Dict[str, Any]:
    """Load checkpoint from given path."""
    
    # Verify checkpoint before loading
    checkpoint_info = _verify_checkpoint_loading(model, checkpoint_path)
    logger.info(f"Checkpoint verification: {checkpoint_info}")


    # Load best model for testing
    if checkpoint_info.get('checkpoint_exists', False) and checkpoint_info.get('can_load', False):
        try:
            best_checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(best_checkpoint['model_state_dict'])
            loaded_epoch = best_checkpoint.get('epoch', 'unknown')
            loaded_metric = best_checkpoint.get('best_primary', 'unknown')
            logger.info(f"✓ Successfully loaded best model from epoch {loaded_epoch} (primary metric: {loaded_metric:.4f})")
        except Exception as e:
            logger.error(f"✗ Failed to load best checkpoint despite verification: {e}")
            logger.info("Using current model state for testing")
    else:
        logger.warning("✗ Best checkpoint not available or cannot be loaded, using current model state")
        if 'error' in checkpoint_info:
            logger.warning(f"Checkpoint error: {checkpoint_info['error']}")

    return model






def _verify_checkpoint_loading(model: nn.Module, checkpoint_path: Path) -> Dict[str, Any]:
    """Verify that checkpoint can be loaded and return info about it."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check what keys are in the checkpoint
        available_keys = list(checkpoint.keys())
        
        # Try to load the model state dict
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            # Check if all model parameters can be loaded
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(model_state.keys())
            
            missing_in_checkpoint = model_keys - checkpoint_keys
            unexpected_in_checkpoint = checkpoint_keys - model_keys
            
            info = {
                'checkpoint_exists': True,
                'checkpoint_keys': available_keys,
                'epoch': checkpoint.get('epoch', 'unknown'),
                'best_primary': checkpoint.get('best_primary', 'unknown'),
                'missing_keys': list(missing_in_checkpoint),
                'unexpected_keys': list(unexpected_in_checkpoint),
                'can_load': len(missing_in_checkpoint) == 0
            }
        else:
            info = {
                'checkpoint_exists': True,
                'checkpoint_keys': available_keys,
                'error': 'model_state_dict not found in checkpoint'
            }
            
        return info
        
    except Exception as e:
        return {
            'checkpoint_exists': False,
            'error': str(e)
        }