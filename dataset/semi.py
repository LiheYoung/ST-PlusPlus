from dataset.transform import crop, hflip, normalize, resize, blur, cutout

import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size, labeled_id_path=None, unlabeled_id_path=None, pseudo_mask_path=None):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        :param labeled_id_path: path of labeled image ids, needed in train or semi_train mode.
        :param unlabeled_id_path: path of unlabeled image ids, needed in semi_train or label mode.
        :param pseudo_mask_path: path of generated pseudo masks, needed in semi_train mode.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        self.pseudo_mask_path = pseudo_mask_path

        if mode == 'semi_train':
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
            self.ids = \
                self.labeled_ids * math.ceil(len(self.unlabeled_ids) / len(self.labeled_ids)) + self.unlabeled_ids

        else:
            if mode == 'val':
                id_path = 'dataset/splits/%s/val.txt' % name
            elif mode == 'label':
                id_path = unlabeled_id_path
            elif mode == 'train':
                id_path = labeled_id_path

            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

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
            fname = os.path.basename(id.split(' ')[1])
            mask = Image.open(os.path.join(self.pseudo_mask_path, fname))

        # basic augmentation on all training images
        base_size = 400 if self.name == 'pascal' else 2048
        img, mask = resize(img, mask, base_size, (0.5, 2.0))
        img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, p=0.5)

        # strong augmentation on unlabeled images
        if self.mode == 'semi_train' and id in self.unlabeled_ids:
            if random.random() < 0.8:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.2)(img)
            img = blur(img, p=0.5)
            img, mask = cutout(img, mask, p=0.5)

        img, mask = normalize(img, mask)

        return img, mask

    def __len__(self):
        return len(self.ids)
