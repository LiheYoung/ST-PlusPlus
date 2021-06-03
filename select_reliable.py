from dataset.cityscapes import Cityscapes
from dataset.pascal import PASCAL
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.pspnet import PSPNet
from util.utils import count_params, meanIOU

import argparse
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Semi-supervised Semantic Segmentation -- Selecting Reliable IDs')

    parser.add_argument('--data-root',
                        type=str,
                        default='/data/lihe/datasets/PASCAL-VOC-2012',
                        help='root path of training dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='pascal',
                        choices=['pascal', 'cityscapes', 'coco'],
                        help='training dataset')
    parser.add_argument('--backbone',
                        type=str,
                        choices=['resnet50', 'resnet101'],
                        default='resnet50',
                        help='backbone of semantic segmentation model')
    parser.add_argument('--model',
                        type=str,
                        default='deeplabv3plus',
                        choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
                        help='model for semantic segmentation')
    parser.add_argument('--unlabeled-id-path',
                        type=str,
                        default=None,
                        required=True,
                        help='path of unlabeled image ids')
    parser.add_argument('--reliable-id-path',
                        type=str,
                        default=None,
                        required=True,
                        help='path of output reliable image ids')

    args = parser.parse_args()
    return args


def compute_reliability(preds, num_classes):
    mIOU = []
    for i in range(len(preds) - 1):
        metric = meanIOU(num_classes=num_classes)
        metric.add_batch(preds[i], preds[-1])
        mIOU.append(metric.evaluate()[-1])
    return sum(mIOU) / len(mIOU)


def select_reliable(dataloader, models, args):
    if not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)

    for i in range(len(models)):
        models[i].eval()
    tbar = tqdm(dataloader)

    id_to_reliability = []

    with torch.no_grad():
        for i, (img, mask, id) in enumerate(tbar):
            img = img.cuda()

            preds = []
            for model in models:
                preds.append(torch.argmax(model(img), dim=1).cpu().numpy())

            reliability = compute_reliability(preds, len(dataloader.dataset.CLASSES))
            id_to_reliability.append((id[0], reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[:len(id_to_reliability) // 2]:
            f.write(elem[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2:]:
            f.write(elem[0] + '\n')


if __name__ == '__main__':
    args = parse_args()
    print(args)

    datasets = {'pascal': PASCAL, 'cityscapes': Cityscapes, 'coco': COCO}

    unlabeled_set = datasets[args.dataset](args.data_root, 'label', None, None, args.unlabeled_id_path)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=1, shuffle=False,
                                  pin_memory=False, num_workers=4, drop_last=False)

    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model1 = model_zoo[args.model](args.backbone, len(unlabeled_set.CLASSES))
    model1.load_state_dict(torch.load(
        'outdir/models/pascal/1_4/split_0/checkpoints/pspnet_resnet50_epoch_19_66.19.pth'),
        strict=True)
    model1 = model1.cuda()

    model2 = model_zoo[args.model](args.backbone, len(unlabeled_set.CLASSES))
    model2.load_state_dict(torch.load(
        'outdir/models/pascal/1_4/split_0/checkpoints/pspnet_resnet50_epoch_49_68.27.pth'),
        strict=True)
    model2 = model2.cuda()

    model3 = model_zoo[args.model](args.backbone, len(unlabeled_set.CLASSES))
    model3.load_state_dict(torch.load(
        'outdir/models/pascal/1_4/split_0/checkpoints/pspnet_resnet50_epoch_79_69.15.pth'),
        strict=True)
    model3 = model3.cuda()

    models = [model1, model2, model3]

    select_reliable(unlabeled_loader, models, args)
