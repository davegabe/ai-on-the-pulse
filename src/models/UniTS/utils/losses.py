# Part of the file is from https://github.com/thuml/Time-Series-Library/blob/main/utils/losses.py
"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch.nn as nn


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class UnifiedMaskRecLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward_mim_loss(self, target, pred, pad_mask):
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        combined_mask = pad_mask.bool()

        loss = (loss * combined_mask).sum() / combined_mask.sum()
        return loss

    def forward(self, outputs, target, pad_mask):
        student_cls, student_fore, _ = outputs

        mask_loss = self.forward_mim_loss(target, student_fore, pad_mask)

        if student_cls is not None:
            cls_loss = self.forward_mim_loss(target, student_cls, pad_mask)
        else:
            cls_loss = 0.0 * mask_loss

        total_loss = dict(cls_loss=cls_loss,
                          mask_loss=mask_loss, loss=mask_loss+cls_loss)
        return total_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        Focal Loss implementation with flexible alpha parameter.
        
        Args:
            alpha: Class weights. Can be:
                - None: No class weighting
                - float: Global weighting factor
                - tensor: Per-class weights
            gamma: Focusing parameter, reduces the relative loss for well-classified examples
                  Higher gamma gives more weight to hard examples
            reduction: 'mean', 'sum' or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Calculate focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
        """
        # Standard cross entropy calculation (without reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Probability of the correct class
        pt = t.exp(-ce_loss)
        
        # Apply focusing parameter gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting based on the type of alpha
        if self.alpha is not None:
            if isinstance(self.alpha, t.Tensor):
                # Get weights for the specific classes in this batch
                batch_alpha = self.alpha[targets]
                focal_loss = batch_alpha * focal_weight * ce_loss
            else:
                # Alpha is a scalar
                focal_loss = self.alpha * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss