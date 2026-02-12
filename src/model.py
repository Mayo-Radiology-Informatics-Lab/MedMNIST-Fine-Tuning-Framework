import timm
import math
import sys
import logging
from typing import List, Dict, Any, Tuple
from torch import nn
import torch

logger = logging.getLogger(__name__)

__all__ = ['init_model', 'log_model_analysis_structure']


def init_model(model_name: str, n_outputs: int, strategy: str, unfreeze_pct: int, direction: str, device: torch.device) -> nn.Module:
    """
    Initialize a model with the given configuration.

    Args:
        model_name: Name of the backbone model (e.g., 'resnet18')
        n_outputs: Number of output classes/values
        strategy: Training strategy ('LP', 'full', 'partial')
        unfreeze_pct: Percentage of layers to unfreeze for partial strategy
        direction: Direction for partial unfreezing ('start' or 'end')
        device: Device to move model to

    Returns:
        Initialized model
    """
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=n_outputs)
        logger.info(f"Created model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to create model {model_name}: {e}")
        sys.exit(1)

    _apply_strategy(
        model,
        strategy,
        unfreeze_pct,
        direction,
    )
    model.to(device)

    return model



def _apply_strategy(
    model: nn.Module,
    strategy: str,
    unfreeze_pct: int = 20,
    direction: str = 'end',
):
    strategy = strategy.lower()
    if strategy == 'partial':
        direction = (direction or 'end').lower()
        if direction not in ('start', 'end'):
            logger.warning(f"Unknown direction '{direction}', defaulting to 'end'")
            direction = 'end'

    extras = ""
    if strategy == 'partial':
        extras = f" with {unfreeze_pct}% ({direction})"
    logger.info(f"Applying strategy: {strategy}{extras}")

    if strategy == 'lp':
        _set_trainable(model, False)
        classifiers = _get_classifier_modules(model)
        trainable_params = 0
        for clf in classifiers:
            for p in clf.parameters():
                p.requires_grad = True
                trainable_params += p.numel()
        logger.info(f"LP strategy: {trainable_params} classifier params trainable")
        return

    if strategy == 'full':
        _set_trainable(model, True)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Full strategy: {total_params} trainable parameters")
        return

    if strategy == 'partial':
        _set_trainable(model, False)

        blocks = _get_encoder_blocks(model)  # List[(name, module)]
        n_blocks = len(blocks)
        if n_blocks == 0:
            logger.warning("No encoder blocks found for partial strategy")
            # still unfreeze classifier
            for clf in _get_classifier_modules(model):
                for p in clf.parameters():
                    p.requires_grad = True
            return

        k = max(1, math.ceil(n_blocks * (unfreeze_pct / 100.0)))
        if direction == 'start':
            indices = list(range(0, k))
        else:  # 'end'
            indices = list(range(n_blocks - k, n_blocks))

        encoder_params = 0
        unfrozen_names = []
        for idx in indices:
            name, block = blocks[idx]
            for p in block.parameters():
                p.requires_grad = True
                encoder_params += p.numel()
            unfrozen_names.append(name)

        # Always unfreeze classifier
        clf_params = 0
        for clf in _get_classifier_modules(model):
            for p in clf.parameters():
                p.requires_grad = True
                clf_params += p.numel()

        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Partial strategy: unfroze {len(indices)}/{n_blocks} blocks from {direction} "
            f"({encoder_params} encoder params) + {clf_params} classifier params. "
            f"Total trainable now: {total_trainable}."
        )
        # Show a short preview of which blocks:
        preview = ", ".join(unfrozen_names[:5]) + (" ..." if len(unfrozen_names) > 5 else "")
        logger.info(f"Unfrozen blocks: {preview}")
        return

    raise ValueError(f"Unknown strategy: {strategy}")

def _set_trainable(model: nn.Module, requires_grad: bool):
    """Set all model parameters to trainable or frozen."""
    for p in model.parameters():
        p.requires_grad = requires_grad


def _get_classifier_modules(model: nn.Module) -> List[nn.Module]:
    """Get classifier/head modules from various model architectures."""
    classifiers = []
    
    # For ResNet: the 'fc' layer
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Module):
        classifiers.append(model.fc)
        logger.debug(f"Found classifier: fc ({type(model.fc).__name__})")
    
    # For other models: common names
    for name in ['classifier', 'head', 'heads']:
        if hasattr(model, name):
            attr = getattr(model, name)
            if isinstance(attr, nn.Module):
                classifiers.append(attr)
                logger.debug(f"Found classifier: {name} ({type(attr).__name__})")
    
    # Try timm's get_classifier method
    if hasattr(model, 'get_classifier') and callable(getattr(model, 'get_classifier')):
        try:
            clf = model.get_classifier()
            if isinstance(clf, nn.Module) and clf not in classifiers:
                classifiers.append(clf)
                logger.debug(f"Found classifier via get_classifier(): {type(clf).__name__}")
        except Exception as e:
            logger.debug(f"get_classifier() failed: {e}")
    
    # Fallback: find Linear layers that could be classifiers
    if not classifiers:
        logger.debug("No obvious classifier found, searching for Linear layers...")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Heuristic: classifier is usually the last linear layer or has "classifier" in name
                if any(clf_name in name.lower() for clf_name in ['fc', 'classifier', 'head']) or len(list(model.named_modules())) - list(model.named_modules()).index((name, module)) <= 3:
                    classifiers.append(module)
                    logger.debug(f"Found potential classifier: {name} ({type(module).__name__}, {sum(p.numel() for p in module.parameters())} params)")
    
    return classifiers





def _get_encoder_blocks(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Return a list of (name, module) blocks in forward order.
    We try to pick architecturally meaningful "blocks" per backbone.
    """
    blocks: List[Tuple[str, nn.Module]] = []

    # ---- Swin: stages (layers) contain transformer "blocks" ----
    if hasattr(model, 'layers'):
        try:
            for i, layer in enumerate(model.layers):
                if hasattr(layer, 'blocks'):
                    for j, blk in enumerate(layer.blocks):
                        blocks.append((f'layers.{i}.blocks.{j}', blk))
        except Exception:
            pass

    # ---- ViT: encoder transformer blocks ----
    if hasattr(model, 'blocks') and isinstance(getattr(model, 'blocks'), (nn.ModuleList, nn.Sequential)):
        for i, blk in enumerate(model.blocks):
            blocks.append((f'blocks.{i}', blk))

    # ---- ResNet: stage modules (coarser granularity) ----
    for name in ['layer1', 'layer2', 'layer3', 'layer4']:
        if hasattr(model, name):
            layer_mod = getattr(model, name)
            blocks.append((name, layer_mod))

    # ---- DenseNet: use high-level DenseBlocks / Transition layers from features ----
    if hasattr(model, 'features'):
        feats = model.features
        # Prefer named DenseBlocks/Transition blocks if present
        dense_like = []
        if hasattr(feats, '_modules') and isinstance(feats._modules, dict):
            for key, mod in feats._modules.items():
                if key.startswith('denseblock') or key.startswith('transition'):
                    dense_like.append((f'features.{key}', mod))
        if dense_like:
            blocks.extend(dense_like)
        else:
            # Fallback: every child of features in order
            if isinstance(feats, (nn.Sequential, nn.ModuleList)):
                for i, mod in enumerate(feats):
                    blocks.append((f'features.{i}', mod))

    # ---- De-duplicate while preserving order (modules can appear in multiple collections) ----
    seen = set()
    unique_blocks: List[Tuple[str, nn.Module]] = []
    for name, mod in blocks:
        mid = id(mod)
        if mid not in seen:
            seen.add(mid)
            unique_blocks.append((name, mod))

    return unique_blocks

def log_model_analysis_structure(model: nn.Module) -> Tuple[int, int]:
    """
    Analyze and log model structure, returning parameter counts.

    Args:
        model: The model to analyze

    Returns:
        Tuple of (total_params, trainable_params)
    """
    model_analysis = _analyze_model_structure(model)
    logger.info("Model analysis:")
    logger.info(f"  Total parameters: {model_analysis['total_params']:,}")
    logger.info(f"  Trainable parameters: {model_analysis['trainable_params']:,}")
    logger.info(f"  Frozen parameters: {model_analysis['frozen_params']:,}")

    if model_analysis['classifier_candidates']:
        logger.info("  Classifier candidates:")
        for clf in model_analysis['classifier_candidates']:
            status = "TRAINABLE" if clf['trainable'] > 0 else "FROZEN"
            logger.info(f"    {clf['name']}: {clf['params']} params ({status})")

    # Double-check with the standard counting function
    trainable_params = _num_trainable_params(model)
    total_params = sum(p.numel() for p in model.parameters())

    if trainable_params != model_analysis['trainable_params']:
        logger.error(f"Parameter count mismatch! Analysis: {model_analysis['trainable_params']}, Direct count: {trainable_params}")

    logger.info(f"Final count - Trainable: {trainable_params:,} / Total: {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    return total_params, trainable_params






def _analyze_model_structure(model: nn.Module) -> Dict[str, Any]:
    """Analyze model structure to understand parameter distribution."""
    analysis = {
        'total_params': 0,
        'trainable_params': 0,
        'frozen_params': 0,
        'modules': {},
        'classifier_candidates': []
    }
    
    # Analyze each named module
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if module_params > 0:
                analysis['modules'][name] = {
                    'type': type(module).__name__,
                    'total_params': module_params,
                    'trainable_params': module_trainable,
                    'frozen_params': module_params - module_trainable
                }
                
                # Check if this could be a classifier
                if any(clf_name in name.lower() for clf_name in ['fc', 'classifier', 'head']):
                    analysis['classifier_candidates'].append({
                        'name': name,
                        'type': type(module).__name__,
                        'params': module_params,
                        'trainable': module_trainable
                    })
    
    # Calculate totals
    analysis['total_params'] = sum(p.numel() for p in model.parameters())
    analysis['trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    analysis['frozen_params'] = analysis['total_params'] - analysis['trainable_params']
    
    return analysis


def _num_trainable_params(model: nn.Module) -> int:
    """Count number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)