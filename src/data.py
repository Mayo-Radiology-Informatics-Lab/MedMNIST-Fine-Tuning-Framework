import logging
from typing import Optional

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import pickle
import numpy as np

logger = logging.getLogger(__name__)

# Default image size (can be overridden)
IMG_SIZE = 224

__all__ = ['PNGFolderDataset', 'prepare_dataloader']

# ---------------------
# Dataset
# ---------------------
class PNGFolderDataset(Dataset):
    """Dataset class for PNG images with pickle labels."""

    def __init__(self, root: Path, transform=None, multilabel: bool = False, regression: bool = False):
        self.root = Path(root)
        self.transform = transform
        self.multilabel = multilabel
        self.regression = regression

        # Load labels
        labels_path = self.root / 'labels.pkl'
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        with open(labels_path, 'rb') as f:
            labels = pickle.load(f)
        self.labels = np.array(labels)

        # Discover images: either in class_* subfolders or flat
        class_dirs = sorted([p for p in self.root.glob('class_*') if p.is_dir()])
        if class_dirs:
            # Sort numerically by suffix for consistent ordering
            class_dirs.sort(key=lambda x: int(x.name.split('_')[1]) if x.name.split('_')[1].isdigit() else 0)
            self.img_paths = []
            for c in class_dirs:
                self.img_paths.extend(sorted(c.glob('*.png')))
        else:
            self.img_paths = sorted(self.root.glob('*.png'))

        if len(self.img_paths) != len(self.labels):
            raise ValueError(f"Mismatch: Labels ({len(self.labels)}) != images ({len(self.img_paths)}) in {self.root}")

        logger.info(f"Loaded {len(self.img_paths)} images from {self.root}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            # Return a black image as fallback
            img = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
            
        if self.transform:
            img = self.transform(img)
            
        y = self.labels[idx]
        if self.regression:
            y = np.array([y], dtype=np.float32)
        elif self.multilabel:
            y = np.array(y, dtype=np.float32)  # shape [L]
        else:
            # For classification, ensure we have a scalar (0D) integer
            if isinstance(y, (list, tuple, np.ndarray)) and len(np.array(y).shape) > 0:
                # If y is an array-like with one element, extract it
                y = np.array(y).flatten()
                if len(y) == 1:
                    y = int(y[0])
                else:
                    # This shouldn't happen for single-label classification
                    logger.warning(f"Multi-element label found for single-label classification: {y}")
                    y = int(y[0])  # Take first element as fallback
            else:
                y = int(y)
        return img, y




def prepare_dataloader(dataset_dir, set_group: str, transforms, task: str, batch_size: int, num_workers: int) -> DataLoader:
    """
    Prepare a DataLoader for the given dataset directory.

    Args:
        dataset_dir: Path to the dataset split directory (e.g., /path/to/dataset/train)
        set_group: One of 'train', 'val', 'test'
        transforms: Torchvision transforms to apply
        task: Task type ('classification', 'multilabel', 'regression')
        batch_size: Batch size for the DataLoader
        num_workers: Number of workers for data loading

    Returns:
        DataLoader instance
    """
    if isinstance(dataset_dir, str):
        dataset_dir = Path(dataset_dir)

    dataset = PNGFolderDataset(
        root=dataset_dir,
        transform=transforms,
        multilabel=(task == 'multilabel'),
        regression=(task == 'regression')
    )

    if set_group == 'train':
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloader