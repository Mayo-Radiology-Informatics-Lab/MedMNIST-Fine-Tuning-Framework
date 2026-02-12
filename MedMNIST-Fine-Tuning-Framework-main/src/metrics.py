import math
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

__all__ = ['Metrics', 'compute_metrics']


# ---------------------
# Metrics per task
# ---------------------
@dataclass
class Metrics:
    primary: float
    acc: Optional[float] = None
    auc: Optional[float] = None
    f1_macro: Optional[float] = None
    auc_macro: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None

def compute_metrics(task: str, y_true: np.ndarray, logits: np.ndarray) -> Metrics:
    """Compute task-specific metrics."""
    try:
        if task == 'classification':
            probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()
            preds = probs.argmax(1)
            
            acc = accuracy_score(y_true, preds)
            
            # AUC computation
            try:
                if probs.shape[1] > 2:
                    auc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
                else:
                    auc = roc_auc_score(y_true, probs[:, 1])
            except Exception as e:
                logger.warning(f"AUC computation failed: {e}")
                auc = np.nan
            
            f1m = f1_score(y_true, preds, average='macro', zero_division=0)
            primary = auc if not np.isnan(auc) else acc
            
            return Metrics(primary=primary, acc=acc, auc=auc, f1_macro=f1m)

        if task == 'multilabel':
            probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
            # Per-label AUC, macro average
            aucs = []
            for j in range(probs.shape[1]):
                yj = y_true[:, j]
                pj = probs[:, j]
                # Only compute AUC when both classes are present
                if len(np.unique(yj)) == 2:
                    try:
                        aucs.append(roc_auc_score(yj, pj))
                    except Exception:
                        pass
            
            auc_macro = float(np.mean(aucs)) if aucs else 0.0
            return Metrics(primary=auc_macro, auc_macro=auc_macro)

        if task == 'regression':
            preds = logits.squeeze(-1) if logits.ndim > 1 else logits
            if y_true.ndim > 1:
                y_true = y_true.squeeze(-1)
            
            mae = mean_absolute_error(y_true, preds)
            rmse = math.sqrt(mean_squared_error(y_true, preds))
            # Use negative MAE as primary for maximization
            return Metrics(primary=-mae, mae=mae, rmse=rmse)

    except Exception as e:
        logger.error(f"Error computing metrics for task {task}: {e}")
        return Metrics(primary=0.0)

    raise ValueError(f"Unknown task: {task}")
