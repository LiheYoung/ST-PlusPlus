from dataset.transform import crop, hflip, normalize, resize, blur

import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms


class PASCAL(Dataset):
    """
    Dataset for PASCAL VOC 2012 and the augmented SBD.
    """
    CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'potted-plant', 'sheep', 'sofa', 'train', 'monitor']

    def __init__(self, root, mode, size, labeled_id_path=None, pseudo_mask_path=None):
        """
        :param root: root path of the PASCAL dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both the labeled and unlabeled images.
                     val: validation, containing 1,449 images.

        :param size: crop size of training images.
        :param labeled_id_path: path of labeled image ids, not needed in validation mode.
        :param pseudo_mask_path: path of generated pseudo masks, only needed in semi_train mode.
        """
        self.mode = mode
        self.size = size

        self.img_path = os.path.join(root, 'JPEGImages')
        self.mask_path = os.path.join(root, 'SegmentationClass')
        self.id_path = os.path.join(root, 'ImageSets')
        self.pseudo_mask_path = pseudo_mask_path

        if mode == 'val':
            with open(os.path.join(self.id_path, 'val.txt'), 'r') as f:
                self.ids = f.read().splitlines()

        elif mode == 'label':
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(os.path.join(self.id_path, 'train_aug.txt'), 'r') as f:
                self.all_ids = f.read().splitlines()
            # the unlabeled ids
            self.ids = list(set(self.all_ids) - set(self.labeled_ids))
            self.ids.sort()

        elif mode == 'train':
            with open(labeled_id_path, 'r') as f:
                self.ids = f.read().splitlines()

        else:
            assert mode == 'semi_train'

            with open(os.path.join(self.id_path, 'train_aug.txt'), 'r') as f:
                self.ids = f.read().splitlines()
            with open(labeled_id_path) as f:
                self.labeled_ids = f.read().splitlines()

            # oversample the labeled images to the approximate size of unlabeled images
            unlabeled_ids = set(self.ids) - set(self.labeled_ids)
            self.ids += self.labeled_ids * (len(unlabeled_ids) // len(self.labeled_ids) - 1)

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id + '.jpg'))

        if self.mode == 'val' or self.mode == 'label':
            mask = Image.open(os.path.join(self.mask_path, id + '.png'))
            img, mask = normalize(img, mask)
            return img, mask, id

        if self.mode == 'train' or (self.mode == 'semi_train' and id in self.labeled_ids):
            mask = Image.open(os.path.join(self.mask_path, id + '.png'))
        else:
            assert self.mode == 'semi_train'
            mask = Image.open(os.path.join(self.pseudo_mask_path, id + '.png'))

        # basic augmentation on all training images
        img, mask = resize(img, mask, (0.5, 2.0))
        img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, p=0.5)

        # strong augmentation on unlabeled images
        if self.mode == 'semi_train' and id not in self.labeled_ids:
            if random.random() < 0.8:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.2)(img)
            img = blur(img, p=0.5)

        img, mask = normalize(img, mask)

        return img, mask

    def __len__(self):
        return len(self.ids)
