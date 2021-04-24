from util.transform import crop, hflip, normalize, resize, blur

import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms


class COCO(Dataset):
    """
    Dataset for MS COCO 2017.
    """
    CLASSES = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
               'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
               'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
               'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
               'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
               'toilet', 'tv', 'laptop',  'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster',  'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self, root, mode, size, labeled_id_path=None, pseudo_mask_path=None):
        """
        :param root: root path of the MS COCO 2017 dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both the labeled and unlabeled images.
                     val: validation, containing 5,000 images.

        :param size: crop size of training images.
        :param labeled_id_path: path of labeled image ids, not needed in validation mode.
        :param pseudo_mask_path: path of generated pseudo masks, only needed in semi_train mode.
        """
        self.mode = mode
        self.size = size

        if mode == 'val':
            self.img_path = os.path.join(root, 'val2017')
        else:
            self.img_path = os.path.join(root, 'train2017')
        self.mask_path = os.path.join(root, 'masks')

        self.pseudo_mask_path = pseudo_mask_path

        if mode == 'val':
            self.ids = os.listdir(self.img_path)
            self.ids = [id.replace('.jpg', '') for id in self.ids if id.endswith('jpg')]
            self.ids.sort()

        elif mode == 'label':
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            self.all_ids = os.listdir(self.img_path)
            self.all_ids = [id.replace('.jpg', '') for id in self.all_ids if id.endswith('jpg')]
            # the unlabeled ids
            self.ids = list(set(self.all_ids) - set(self.labeled_ids))

        elif mode == 'train':
            with open(os.path.join(labeled_id_path), 'r') as f:
                self.ids = f.read().splitlines()

        else:
            # mode == 'semi_train'
            self.ids = os.listdir(self.img_path)
            self.ids = [id.replace('.jpg', '') for id in self.ids if id.endswith('jpg')]
            with open(labeled_id_path) as f:
                self.labeled_ids = f.read().splitlines()

            # oversample the labeled images to the approximate size of unlabeled images
            unlabeled_ids = set(self.ids) - set(self.labeled_ids)
            self.ids += self.labeled_ids * (len(unlabeled_ids) // len(self.labeled_ids) - 1)

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id + '.jpg')).convert('RGB')

        if self.mode == 'val' or self.mode == 'label':
            mask = Image.open(os.path.join(self.mask_path, id + '.png'))
            img, mask = normalize(img, mask)
            return img, mask, id

        if self.mode == 'train' or (self.mode == 'semi_train' and id in self.labeled_ids):
            mask = Image.open(os.path.join(self.mask_path, id + '.png'))
        else:
            # mode == 'semi_train' and the id corresponds to unlabeled image
            mask = Image.open(os.path.join(self.pseudo_mask_path, id + '.png'))

        # basic augmentation on all training images
        img, mask = resize(img, mask, 400, (0.5, 2.0))
        img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, p=0.5)

        # strong augmentation on unlabeled images
        if self.mode == 'semi_train' and id not in self.labeled_ids:
            if random.random() < 0.8:
                img = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)(img)
            img = transforms.RandomGrayscale(p=0.2)(img)
            img = blur(img, p=0.5)

        img, mask = normalize(img, mask)

        return img, mask

    def __len__(self):
        return len(self.ids)
