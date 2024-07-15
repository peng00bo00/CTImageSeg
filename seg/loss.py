import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp

class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = p * targets + (1 - p) * (1 - targets)
        focal_loss = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.sum() / inputs.size(0)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def create_loss_fn(loss_type: str, params: dict):
    """A helper function to create loss function from SMP.
    """

    if loss_type == "FocalLoss":
        params.update("mode", "binary")
        return smp.losses.FocalLoss(**params)
    elif loss_type == "LovaszLoss":
        params.update("mode", "binary")
        params.update("from_logits", True)
        return smp.losses.LovaszLoss(**params)
    else:
        raise NotImplementedError(f"Invalid SMP loss type: {loss_type}")