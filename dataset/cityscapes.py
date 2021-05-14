from util.transform import crop, hflip, normalize, resize, blur, cutout

import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms


class Cityscapes(Dataset):
    """
    Dataset for Cityscapes without coarse masks.
    """
    CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
               'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
               'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    def __init__(self, root, mode, size, labeled_id_path=None, pseudo_mask_path=None):
        """
        :param root: root path of the Cityscapes dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both the labeled and unlabeled images.
                     val: validation, containing 500 images.

        :param size: crop size of training images.
        :param labeled_id_path: path of labeled image ids, not needed in validation mode.
        :param pseudo_mask_path: path of generated pseudo masks, only needed in semi_train mode.
        """
        self.root = root
        self.mode = mode
        self.size = size

        self.pseudo_mask_path = pseudo_mask_path

        if mode == 'val':
            with open(os.path.join(root, 'val.list'), 'r') as f:
                self.ids = f.read().splitlines()

        elif mode == 'label':
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(os.path.join(root, 'train.list'), 'r') as f:
                self.all_ids = f.read().splitlines()
            # the unlabeled ids
            self.ids = list(set(self.all_ids) - set(self.labeled_ids))
            self.ids.sort()

        elif mode == 'train':
            with open(labeled_id_path, 'r') as f:
                self.ids = f.read().splitlines()

        else:
            # mode == 'semi_train'
            with open(os.path.join(root, 'train.list'), 'r') as f:
                self.ids = f.read().splitlines()
            with open(labeled_id_path) as f:
                self.labeled_ids = f.read().splitlines()

            # oversample the labeled images to the approximate size of unlabeled images
            unlabeled_ids = set(self.ids) - set(self.labeled_ids)
            self.ids += self.labeled_ids * math.ceil(len(unlabeled_ids) / len(self.labeled_ids) - 1)

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0]))

        if self.mode == 'val' or self.mode == 'label':
            mask = Image.open(os.path.join(self.root, id.split(' ')[1]))
            img, mask = normalize(img, mask)
            return img, mask, id

        if self.mode == 'train' or (self.mode == 'semi_train' and id in self.labeled_ids):
            mask = Image.open(os.path.join(self.root, id.split(' ')[1]))
        else:
            # mode == 'semi_train' and the id corresponds to unlabeled image
            fname = os.path.basename(id.split(' ')[0]).replace('_leftImg8bit', '')
            mask = Image.open(os.path.join(self.pseudo_mask_path, fname))

        # basic augmentation on all training images
        img, mask = resize(img, mask, 2048, (0.5, 2.0))
        img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, p=0.5)

        # strong augmentation on unlabeled images
        if self.mode == 'semi_train' and id not in self.labeled_ids:
            if random.random() < 0.8:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.2)(img)
            img = blur(img, p=0.5)
            img, mask = cutout(img, mask)

        img, mask = normalize(img, mask)

        return img, mask

    def __len__(self):
        return len(self.ids)
