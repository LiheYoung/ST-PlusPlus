from model.backbone.resnet import resnet50, resnet101

from torch import nn
import torch.nn.functional as F


class BaseNet(nn.Module):
    def __init__(self, backbone):
        super(BaseNet, self).__init__()
        backbone_zoo = {'resnet50': resnet50, 'resnet101': resnet101}
        self.backbone = backbone_zoo[backbone](pretrained=True)

    def base_forward(self, x):
        h, w = x.shape[-2:]

        x = self.backbone.base_forward(x)[-1]
        x = self.head(x)
        x = F.interpolate(x, (h, w), mode="bilinear", align_corners=True)

        return x

    def forward(self, x, tta=False):
        if not tta:
            return self.base_forward(x)

        else:
            h, w = x.shape[-2:]
            scales = [0.5, 0.75, 1.0, 1.5, 2.0]

            final_result = None

            for scale in scales:
                cur_h, cur_w = int(h * scale), int(w * scale)
                cur_x = F.interpolate(x, size=(cur_h, cur_w), mode='bilinear', align_corners=True)

                out = F.softmax(self.base_forward(cur_x), dim=1)
                out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
                final_result = out if final_result is None else (final_result + out)

                out = F.softmax(self.base_forward(cur_x.flip(3)), dim=1).flip(3)
                out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
                final_result += out

            return final_result
