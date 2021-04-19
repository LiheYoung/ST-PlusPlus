from dataset.pascal import PASCAL
from model.deeplabv3plus import DeepLabV3Plus
from util.utils import count_params, meanIOU

import argparse
import os
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Semi-supervised Semantic Segmentation -- Training')

    parser.add_argument('--data-root',
                        type=str,
                        default='/data/lihe/datasets/PASCAL-VOC-2012',
                        help='root path of training dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='pascal',
                        choices=['pascal'],
                        help='training dataset')
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='batch size of training')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate')
    parser.add_argument('--epochs',
                        type=int,
                        default=80,
                        help='training epochs')
    parser.add_argument('--crop-size',
                        type=int,
                        default=321,
                        help='cropping size of training samples')
    parser.add_argument('--backbone',
                        type=str,
                        choices=['resnet50', 'resnet101'],
                        default='resnet50',
                        help='backbone of semantic segmentation model')
    parser.add_argument('--model',
                        type=str,
                        default='deeplabv3plus',
                        help='model for semantic segmentation')
    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train', 'semi_train'],
                        help='choose supervised/semi-supervised setting')
    parser.add_argument('--labeled-id-path',
                        type=str,
                        default='/data/lihe/datasets/PASCAL-VOC-2012/ImageSets/train.txt',
                        help='path of labeled image ids')
    parser.add_argument('--pseudo-mask-path',
                        type=str,
                        default='outdir/pseudo_masks',
                        help='path of generated pseudo masks')
    parser.add_argument('--save-path',
                        type=str,
                        default=None,
                        required=True,
                        help='path of saved checkpoints')

    args = parser.parse_args()
    return args


def main(args):
    save_path = args.save_path if args.save_path else 'outdir/models/%s' % args.dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.dataset == 'pascal':
        trainset = PASCAL(args.data_root, args.mode, args.crop_size, args.labeled_id_path)
        valset = PASCAL(args.data_root, 'val', None)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)
    valloader = DataLoader(valset, batch_size=1, shuffle=False,
                           pin_memory=True, num_workers=16, drop_last=False)

    if args.model == 'deeplabv3plus':
        model = DeepLabV3Plus(args.backbone, len(trainset.CLASSES))
    print('\nParams: %.1fM' % count_params(model))

    criterion = CrossEntropyLoss(ignore_index=255)
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
                     {'params': [param for name, param in model.named_parameters()
                                 if 'backbone' not in name],
                      'lr': args.lr * 10.0}],
                    lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model = DataParallel(model).cuda()

    iters = 0
    total_iters = len(trainloader) * args.epochs

    previous_best = 0.0

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()
        total_loss = 0.0
        tbar = tqdm(trainloader)

        for i, (img, mask) in enumerate(tbar):
            img, mask = img.cuda(), mask.cuda()

            pred = model(img)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0

            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))

        metric = meanIOU(num_classes=len(valloader.dataset.CLASSES))

        model.eval()
        tbar = tqdm(valloader)

        with torch.no_grad():
            for img, mask, _ in tbar:
                img = img.cuda()
                pred = model(img)
                pred = torch.argmax(pred, dim=1)

                metric.add_batch(pred.cpu().numpy(), mask.numpy())
                mIOU = metric.evaluate()[-1]

                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

        mIOU *= 100.0
        if mIOU > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
            previous_best = mIOU
            torch.save(model.module.state_dict(),
                       os.path.join(save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))


if __name__ == '__main__':
    args = parse_args()
    print(args)

    main(args)
