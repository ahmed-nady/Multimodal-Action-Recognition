"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
#https://github.com/rohan-paul/MachineLearning-DeepLearning-Code-for-my-YouTube-Channel/blob/master/PyTorch-Techniques/46-Label-Smoothing-CrossEntropyLoss-Scratch.ipynb
class LabelSmoothingLoss(torch.nn.Module):
    """
    Label smoothing loss implementation.

    Args:
        epsilon (float): Smoothing factor.
        reduction (str): Type of reduction to apply to the loss ('mean', 'sum', or 'none').
        weight (torch.Tensor, optional): Weight tensor for the loss.

    Attributes:
        epsilon (float): Smoothing factor.
        reduction (str): Type of reduction to apply to the loss ('mean', 'sum', or 'none').
        weight (torch.Tensor or None): Weight tensor for the loss.

    Note:
        This implementation assumes the input `predict_tensor` is log probabilities.

    References:
        - Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).
          Rethinking the inception architecture for computer vision. In Proceedings
          of the IEEE conference on computer vision and pattern recognition (pp. 2818-2826).

    """

    def __init__(self, epsilon: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        """
        Reduce the loss tensor based on the specified reduction type.

        Args:
            loss (torch.Tensor): Loss tensor to be reduced.

        Returns:
            torch.Tensor: Reduced loss tensor.

        """
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def linear_combination(self, i, j):
        """
        Perform a linear combination of two tensors.

        Args:
            i (torch.Tensor): First tensor.
            j (torch.Tensor): Second tensor.

        Returns:
            torch.Tensor: Linear combination of the two tensors.

        """
        return (1 - self.epsilon) * i + self.epsilon * j

    def forward(self, predict_tensor, target):
        """
        Forward pass of the label smoothing loss.

        Args:
            predict_tensor (torch.Tensor): Predicted tensor from the model.
            target (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Loss value.

        """
        assert 0 <= self.epsilon < 1

        if self.weight is not None:
            self.weight = self.weight.to(predict_tensor.device)

        num_classes = predict_tensor.size(-1)

        log_preds = F.log_softmax(predict_tensor, dim=-1)

        loss = self.reduce_loss(-log_preds.sum(dim=-1))

        negative_log_likelihood_loss = F.nll_loss(
            log_preds, target.argmax(dim=-1), reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(negative_log_likelihood_loss, loss / num_classes, )
class UnifiedContrastive(nn.Module):
    def __init__(self, reduction='mean'):
        super(UnifiedContrastive, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        sum_neg = ((1 - y_true) * torch.exp(y_pred)).sum(1)
        sum_pos = (y_true * torch.exp(-y_pred)).sum(1)
        loss = torch.log(1 + sum_neg * sum_pos)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss