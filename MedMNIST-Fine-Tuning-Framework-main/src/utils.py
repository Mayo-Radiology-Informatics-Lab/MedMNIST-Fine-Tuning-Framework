from pathlib import Path
import json
import logging
from typing import Any, Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Dataset configuration
REGRESSION_DATASETS = {'bonemnist-reg', 'organamnist-reg'}

CLASS_MULTILABEL = {
    'chestmnist': True,
}

# Number of classes for each dataset (defaults)
DATASET_N_CLASSES = {
    'chestmnist': 14,
    'breastmnist': 2,
    'tissuemnist': 8,
    'organsmnist': 11,
    'retinamnist': 5,
    'pathmnist': 9,
    'dermamnist': 7,
    'octmnist': 4,
    'pneumoniamnist': 2,
    'organamnist': 11,
}

__all__ = [
    'load_training_info',
    'get_task_output_from_ds_name',
    'plot_history_png',
]


def load_training_info(ds_dir: Path) -> Dict[str, Any]:
    """Load training info from JSON file with fallbacks."""
    info_path = ds_dir / 'training_info.json'
    if info_path.exists():
        try:
            with open(info_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load training info: {e}")
    return {}


def get_task_output_from_ds_name(ds_name: str) -> Tuple[str, int]:
    """
    Determine task type and number of outputs from dataset name.

    Args:
        ds_name: Dataset name (lowercase)

    Returns:
        Tuple of (task_type, n_outputs)
    """
    ds_name_lower = ds_name.lower()

    if ds_name_lower in REGRESSION_DATASETS:
        task = 'regression'
        n_outputs = 1
    elif CLASS_MULTILABEL.get(ds_name_lower, False):
        task = 'multilabel'
        n_outputs = DATASET_N_CLASSES.get(ds_name_lower, 14)
    else:
        task = 'classification'
        n_outputs = DATASET_N_CLASSES.get(ds_name_lower, 2)
        if n_outputs <= 1:
            n_outputs = 2

    return task, n_outputs




def plot_history_png(out_dir: Path, history: List[Dict[str, Any]], task: str):
    if not history:
        return
    df = pd.DataFrame(history)

    # Loss curves
    plt.figure()
    if 'train_loss' in df: plt.plot(df['epoch'], df['train_loss'], label='train_loss')
    if 'val_loss' in df:   plt.plot(df['epoch'], df['val_loss'], label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss')
    plt.tight_layout()
    plt.savefig(out_dir / 'loss_curves.png', dpi=200)
    plt.close()

    # Primary metric (always exists)
    if 'primary_metric' in df:
        plt.figure()
        plt.plot(df['epoch'], df['primary_metric'], label='val_primary')
        plt.xlabel('Epoch'); plt.ylabel('Primary (higher=better)')
        plt.legend(); plt.title('Validation Primary')
        plt.tight_layout()
        plt.savefig(out_dir / 'val_primary.png', dpi=200)
        plt.close()

    # Task-specific nice-to-haves
    if task == 'classification' and 'val_auc' in df:
        plt.figure()
        plt.plot(df['epoch'], df['val_auc'], label='val_auc')
        if 'val_acc' in df: plt.plot(df['epoch'], df['val_acc'], label='val_acc')
        if 'val_f1_macro' in df: plt.plot(df['epoch'], df['val_f1_macro'], label='val_f1_macro')
        plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend(); plt.title('Validation Metrics')
        plt.tight_layout()
        plt.savefig(out_dir / 'val_metrics_classification.png', dpi=200)
        plt.close()

    if task == 'multilabel' and 'val_auc_macro' in df:
        plt.figure()
        plt.plot(df['epoch'], df['val_auc_macro'], label='val_auc_macro')
        plt.xlabel('Epoch'); plt.ylabel('AUC (macro)'); plt.legend(); plt.title('Validation AUC Macro')
        plt.tight_layout()
        plt.savefig(out_dir / 'val_metrics_multilabel.png', dpi=200)
        plt.close()

    if task == 'regression' and 'val_mae' in df:
        plt.figure()
        plt.plot(df['epoch'], df['val_mae'], label='val_mae')
        if 'val_rmse' in df: plt.plot(df['epoch'], df['val_rmse'], label='val_rmse')
        plt.xlabel('Epoch'); plt.ylabel('Error'); plt.legend(); plt.title('Validation Errors')
        plt.tight_layout()
        plt.savefig(out_dir / 'val_metrics_regression.png', dpi=200)
        plt.close()
