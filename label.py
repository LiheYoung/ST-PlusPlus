from dataset.cityscapes import Cityscapes
from dataset.coco import COCO
from dataset.pascal import PASCAL
from model.deeplabv3plus import DeepLabV3Plus
from util.utils import color_map, count_params, meanIOU

import argparse
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Semi-supervised Semantic Segmentation -- Pseudo Labeling')

    parser.add_argument('--data-root',
                        type=str,
                        default='/data/lihe/datasets/PASCAL-VOC-2012',
                        help='root path of training dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='pascal',
                        choices=['pascal', 'cityscapes'],
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
                        help='whether to use tta (multi-scale testing and horizontal fliping)')
    parser.add_argument('--labeled-id-path',
                        type=str,
                        default=None,
                        required=True,
                        help='path of labeled image ids')
    parser.add_argument('--pseudo-mask-path',
                        type=str,
                        default=None,
                        required=True,
                        help='path of generated pseudo masks')
    parser.add_argument('--vis-path',
                        type=str,
                        default=None,
                        help='visualize the pseudo masks along with original images and gt masks')

    args = parser.parse_args()
    return args


def label(dataloader, model, args):
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if not os.path.exists(args.vis_path):
        os.makedirs(args.vis_path)

    model.eval()
    tbar = tqdm(dataloader)

    metric = meanIOU(num_classes=len(dataloader.dataset.CLASSES))
    cmap = color_map(args.dataset)

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()
            pred = model(img, args.tta)
            pred = torch.argmax(pred, dim=1)

            metric.add_batch(pred.cpu().numpy(), mask.numpy())
            mIOU = metric.evaluate()[-1]

            pred = Image.fromarray(pred.squeeze(0).cpu().numpy().astype(np.uint8), mode='P')
            pred.putpalette(cmap)

            if args.dataset == 'pascal':
                pred.save('%s/%s.png' % (args.pseudo_mask_path, id[0]))
            elif args.dataset == 'cityscapes':
                fname = os.path.basename(id[0].split(' ')[0]).replace('_leftImg8bit.png', '')
                pred.save('%s/%s.png' % (args.pseudo_mask_path, fname))

            if args.vis_path is not None:
                if args.dataset == 'pascal':
                    img = Image.open(os.path.join(args.data_root, 'JPEGImages', id[0] + '.jpg'))
                    mask = Image.open(os.path.join(args.data_root, 'SegmentationClass', id[0] + '.png'))
                elif args.dataset == 'cityscapes':
                    img = Image.open(os.path.join(args.data_root, id[0].split(' ')[0]))
                    mask = Image.open(os.path.join(args.data_root, id[0].split(' ')[1])).convert('P')
                    mask.putpalette(cmap)

                images = [img, mask, pred]

                widths, heights = zip(*(i.size for i in images))

                total_width = sum(widths) + 40
                max_height = max(heights)

                visualize = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))

                x_offset = 0
                for image in images:
                    visualize.paste(image, (x_offset, 0))
                    x_offset += image.size[0] + 20

                if args.dataset == 'pascal':
                    visualize.save(os.path.join(args.vis_path, id[0] + '.jpg'))
                elif args.dataset == 'cityscapes':
                    visualize.save(os.path.join(args.vis_path, fname + '.jpg'))

            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))


if __name__ == '__main__':
    args = parse_args()
    print(args)

    if args.dataset == 'pascal':
        labelset = PASCAL(args.data_root, 'label', None, args.labeled_id_path)

    elif args.dataset == 'cityscapes':
        labelset = Cityscapes(args.data_root, 'label', None, args.labeled_id_path)

    labelloader = DataLoader(labelset, batch_size=1, shuffle=False,
                             pin_memory=True, num_workers=16, drop_last=False)

    if args.model == 'deeplabv3plus':
        model = DeepLabV3Plus(args.backbone, len(labelset.CLASSES))
    print('\nParams: %.1fM\n' % count_params(model))

    model.load_state_dict(torch.load(args.load_from), strict=True)
    model = model.cuda()

    label(labelloader, model, args)
