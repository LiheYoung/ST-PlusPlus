from model.semseg.base import BaseNet

import torch
from torch import nn
import torch.nn.functional as F


class DeepLabV2(BaseNet):
    def __init__(self, backbone, nclass):
        super(DeepLabV2, self).__init__(backbone)

        self.aspp1 = ASPPConv(self.backbone.channels[-1], nclass, 6)
        self.aspp2 = ASPPConv(self.backbone.channels[-1], nclass, 12)
        self.aspp3 = ASPPConv(self.backbone.channels[-1], nclass, 18)
        self.aspp4 = ASPPConv(self.backbone.channels[-1], nclass, 24)

    def base_forward(self, x):
        h, w = x.shape[-2:]

        out = self.backbone.base_forward(x)[-1]

        out1 = self.aspp1(out)
        out2 = self.aspp2(out)
        out3 = self.aspp3(out)
        out4 = self.aspp4(out)

        out = out1 + out2 + out3 + out4

        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return out


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block
