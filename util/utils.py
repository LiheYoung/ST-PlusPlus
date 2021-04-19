import numpy as np
from PIL import Image


def color_map(n_color=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((n_color, 3), dtype=dtype)
    for i in range(n_color):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist) + 1e-5)
        return iu, np.nanmean(iu)
