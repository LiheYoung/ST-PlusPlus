from dataset.transform import crop, hflip, normalize, resize

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PASCAL(Dataset):
    CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'potted-plant', 'sheep', 'sofa', 'train', 'monitor']

    def __init__(self, root, mode, size):
        self.mode = mode
        self.size = size

        self.img_path = os.path.join(root, 'JPEGImages')
        self.mask_path = os.path.join(root, 'SegmentationClass')
        self.id_path = os.path.join(root, 'ImageSets')

        id_filename = 'train' if mode == 'train' else mode
        with open(os.path.join(self.id_path, '%s.txt' % id_filename), 'r') as f:
            self.ids = f.read().splitlines()

        if self.mode == 'train':
            self.colorjitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)

    def __getitem__(self, item):
        id_ = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id_ + '.jpg'))
        mask = Image.open(os.path.join(self.mask_path, id_ + '.png'))

        if self.mode == 'train':
            img, mask = resize(img, mask, (0.5, 2.0))
            img, mask = crop(img, mask, self.size)
            img, mask = hflip(img, mask)
            img = self.colorjitter(img)
        img, mask = normalize(img, mask)

        return img, mask

    def __len__(self):
        return len(self.ids)
