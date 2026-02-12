import torch
import random
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ['init_experiment']

def init_experiment(args):
    """Initialize experiment: set seed, create output directory, return device."""

    seed_everything(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Get dataset name for run naming
    ds_name = args.dataset.lower()

    strategy_lower = args.strategy.lower()
    if strategy_lower == 'partial':
        run_name = f"{ds_name}_{args.backbone}_{args.strategy}_pct{args.unfreeze_pct}_dir{args.direction}_seed{args.seed}"
    else:
        run_name = f"{ds_name}_{args.backbone}_{args.strategy}_pct0_seed{args.seed}"

    output_dir = Path(args.out_dir) / run_name
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / 'checkpoints' / 'best.pt'

    return output_dir, checkpoint_path, device


def seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set all random seeds to {seed}")