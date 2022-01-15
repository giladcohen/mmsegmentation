import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .aspp_head import ASPPHead
from mmcv.runner import force_fp32


@HEADS.register_module()
class ASPPHeadEmb(ASPPHead):
    def __init__(self, dilations=(1, 6, 12, 18), **kwargs):
        self.glove_dim = kwargs.pop('glove_dim')
        super().__init__(dilations, **kwargs)
        self.glove_conv1 = nn.Conv2d(self.channels, self.glove_dim, kernel_size=1)
        self.glove_conv2 = nn.Conv2d(self.glove_dim, self.glove_dim, kernel_size=1)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.glove_conv1(feat)
        output = self.glove_conv2(output)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        for loss_decode in self.loss_decode:
            # debug
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        return loss
