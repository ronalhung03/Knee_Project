from __future__ import annotations

import torch
import torch.nn.functional as F

from .constants import NUM_CLASSES

try:
    from coral_pytorch.dataset import corn_label_from_logits as _corn_label_from_logits
    from coral_pytorch.losses import corn_loss as _corn_loss
except ModuleNotFoundError:
    _corn_label_from_logits = None
    _corn_loss = None


def corn_label_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if _corn_label_from_logits is not None:
        return _corn_label_from_logits(logits)
    return torch.sum(torch.sigmoid(logits) > 0.5, dim=1)


def corn_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    if _corn_loss is not None:
        return _corn_loss(logits, labels, num_classes)

    losses = []
    for task_idx in range(num_classes - 1):
        if task_idx == 0:
            mask = torch.ones_like(labels, dtype=torch.bool)
        else:
            mask = labels > (task_idx - 1)
        if not torch.any(mask):
            continue
        binary_labels = (labels[mask] > task_idx).float()
        losses.append(F.binary_cross_entropy_with_logits(logits[mask, task_idx], binary_labels))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()

