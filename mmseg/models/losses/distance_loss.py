"""Implementing L1, L2, and Linf distance losses"""
import os
import numpy as np
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import L1Loss

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss
from research.losses.losses import L2Loss, LinfLoss, CosineEmbeddingLossV2

@LOSSES.register_module()
class DistanceLoss(nn.Module):
    """Distance loss: L1, L2, Linf, or Cosine Similarity.

    Args:
        loss_type (str): L1/L2/Linf/cosine.
        per_image (bool, optional): If per_image is True, compute the loss per
            image instead of per batch. Default: False.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_lovasz'.
    """

    @staticmethod
    def get_idx_to_vec(path):
        assert os.path.exists(path), "path to 'idx_to_vec' ({}) does not exist".format(path)
        return np.load(path)

    def __init__(self, loss_type, idx_to_vec_path):
        super().__init__()
        assert loss_type in ('L1', 'L2', 'Linf', 'cosine'), "loss_type should be 'L1/L2/Linf' or 'multi_class'."

        if loss_type == 'L1':
            self.dist_criterion = L1Loss()
        elif loss_type == 'L2':
            self.dist_criterion = L2Loss()
        elif loss_type == 'Linf':
            self.dist_criterion = LinfLoss()
        else:
            self.dist_criterion = CosineEmbeddingLossV2()
        self.idx_to_vec = torch.from_numpy(self.get_idx_to_vec(idx_to_vec_path))
        self.ignore_index = 255
        self.cnt = 0

        self._loss_name = 'loss_' + loss_type

    def targets_to_embs(self, targets):
        return self.idx_to_vec[targets.cpu().long()].to(targets.device)

    def flatten_embs(self, embs, labels):
        B, C, H, W = embs.size()
        embs = embs.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.view(-1)
        if self.ignore_index is None:
            return embs, labels
        valid = (labels != self.ignore_index)
        vembs = embs[valid.nonzero().squeeze()]
        vlabels = labels[valid]
        return vembs, vlabels

    def forward(self, embs, labels, **kwargs):
        embs, labels = self.flatten_embs(embs, labels)
        embs_gt = self.targets_to_embs(labels)
        np.save('/home/gilad/tmp/debug/embs_{}'.format(self.cnt), embs)
        np.save('/home/gilad/tmp/debug/labels_{}'.format(self.cnt), labels)
        np.save('/home/gilad/tmp/debug/embs_gt_{}'.format(self.cnt), embs_gt)
        self.cnt += 1
        loss = self.dist_criterion(embs, embs_gt)
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
