from dataset.pascal import PASCAL
from model.deeplabv3plus import DeepLabV3Plus
from util.metric import meanIOU
from util.params import count_params

import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Semi-supervised Semantic Segmentation -- Validation')

    parser.add_argument('--data-root',
                        type=str,
                        default='/data/lihe/datasets/PASCAL-VOC-2012',
                        help='root path of training dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='pascal',
                        choices=['pascal'],
                        help='training dataset')
    parser.add_argument('--backbone',
                        type=str,
                        choices=['resnet50', 'resnet101'],
                        default='resnet50',
                        help='backbone of semantic segmentation model')
    parser.add_argument('--model',
                        type=str,
                        default='deeplabv3plus',
                        help='model for semantic segmentation')
    parser.add_argument('--load-from',
                        type=str,
                        default='',
                        help='path of trained model')
    parser.add_argument('--tta',
                        dest='tta',
                        default=False,
                        action='store_true',
                        help='whether to use tta(multi-scale testing and horizontal fliping)')

    args = parser.parse_args()
    return args


def evaluation(dataloader, model, tta=False):
    metric = meanIOU(num_classes=len(dataloader.dataset.CLASSES))

    model.eval()
    tbar = tqdm(dataloader)

    with torch.no_grad():
        for img, mask in tbar:
            img = img.cuda()
            pred = model(img, tta)
            pred = torch.argmax(pred, dim=1)

            metric.add_batch(pred.cpu().numpy(), mask.numpy())
            mIOU = metric.evaluate()[-1]

            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

    return mIOU


if __name__ == '__main__':
    args = parse_args()

    if args.dataset == 'pascal':
        valset = PASCAL(args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=1, shuffle=False,
                           pin_memory=True, num_workers=16, drop_last=False)

    if args.model == 'deeplabv3plus':
        model = DeepLabV3Plus(args.backbone, len(valset.CLASSES))
    print('\nParams: %.1fM\n' % count_params(model))

    model.load_state_dict(torch.load(args.load_from), strict=True)
    model = model.cuda()

    evaluation(valloader, model, args.tta)
