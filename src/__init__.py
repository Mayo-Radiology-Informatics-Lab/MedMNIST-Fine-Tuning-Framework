"""
MedMNIST Fine-Tuning Trainer - Source modules

This package contains the following modules:
- experiment: Experiment initialization (seed, output dirs, device)
- data: Dataset and DataLoader utilities
- transforms: Image augmentation transforms
- utils: Utility functions (loading info, plotting)
- model: Model initialization and strategy application
- optimizer: Optimizer and scheduler initialization
- callbacks: Training callbacks (early stopping, checkpointing)
- metrics: Metric computation for different tasks
"""

from .experiment import init_experiment
from .data import PNGFolderDataset, prepare_dataloader
from .transforms import create_transforms
from .utils import load_training_info, get_task_output_from_ds_name, plot_history_png
from .model import init_model, log_model_analysis_structure
from .optimizer import init_optimizer_scheduler
from .callbacks import EarlyStopping, save_checkpoint, load_checkpoint
from .metrics import Metrics, compute_metrics

__all__ = [
    # experiment
    'init_experiment',
    # data
    'PNGFolderDataset',
    'prepare_dataloader',
    # transforms
    'create_transforms',
    # utils
    'load_training_info',
    'get_task_output_from_ds_name',
    'plot_history_png',
    # model
    'init_model',
    'log_model_analysis_structure',
    # optimizer
    'init_optimizer_scheduler',
    # callbacks
    'EarlyStopping',
    'save_checkpoint',
    'load_checkpoint',
    # metrics
    'Metrics',
    'compute_metrics',
]
