import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from aspp_head import ASPPHead


@HEADS.register_module()
class ASPPHeadExt(ASPPHead):
    def __init__(self, dilations=(1, 6, 12, 18), **kwargs):
        super().__init__(dilations, **kwargs)
        self.glove_conv = nn.Conv2d(kwargs['channels'], kwargs['glove_dim'], kernel_size=1)
        self.glove_conv_seg = nn.Conv2d(kwargs['glove_dim'], kwargs['num_classes'], kernel_size=1)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.glove_conv(feat)
        output = self.glove_conv_seg(output)
        return output
