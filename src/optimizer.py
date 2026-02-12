import logging
import torch

logger = logging.getLogger(__name__)

__all__ = ['init_optimizer_scheduler']



def init_optimizer_scheduler(model: torch.nn.Module, lr_classifier: float, lr_encoder: float, strategy: str, weight_decay: float, scheduler: str, epochs: int):



    if lr_encoder is not None and strategy != 'lp':
        # Separate learning rates for encoder and classifier
        classifier_names = {'classifier', 'fc', 'head', 'heads'}
        encoder_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(cls_name in name for cls_name in classifier_names):
                classifier_params.append(param)
            else:
                encoder_params.append(param)
        
        param_groups = [
            {'params': encoder_params, 'lr': lr_encoder},
            {'params': classifier_params, 'lr': lr_classifier}
        ]
        logger.info(f"Using separate LRs: encoder={lr_encoder}, classifier={lr_classifier}")
    else:
        param_groups = [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': lr_classifier}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    # Scheduler
    if scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, epochs // 4), T_mult=2)
    elif scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
    elif scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)

    return optimizer, scheduler