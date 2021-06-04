from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils import count_params, meanIOU

import argparse
from copy import deepcopy
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm


MODE = None


def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['pascal', 'cityscapes'], default='pascal')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--crop-size', type=int, default=None,)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet50')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
                        default='deeplabv3plus')

    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--pseudo-mask-path', type=str, required=True)

    parser.add_argument('--save-path', type=str, required=True)

    # arguments for ST++
    parser.add_argument('--reliable-id-path', type=str)
    parser.add_argument('--plus', dest='plus', default=False, action='store_true',
                        help='whether to use ST++')

    args = parser.parse_args()
    return args


def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)

    criterion = CrossEntropyLoss(ignore_index=255)

    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=args.batch_size if args.dataset == 'cityscapes' else 1,
                           shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    # <====================== Supervised training with labeled images (SupOnly) ======================>
    print('<====================== Supervised training with labeled images (SupOnly)')

    MODE = 'train'

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, args.labeled_id_path)
    trainset.ids = 2 * trainset.ids if len(trainset.ids) < 200 else trainset.ids
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems(args)

    checkpoints = train(model, trainloader, valloader, criterion, optimizer, args)

    if not args.plus:
        # <========================= Pseudo label all unlabeled images =========================>
        print('================> Pseudo labeling all unlabeled images')

        dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

        label(model, dataloader, args)

        # <==================== Re-training on labeled and unlabeled images ====================>
        print('================> Re-training on labeled and unlabeled images')

        MODE = 'semi_train'

        trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                               args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path)
        trainloader = DataLoader(trainset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

        model, optimizer = init_basic_elems(args)

        train(model, trainloader, valloader, criterion, optimizer, args)

        return

    # <================================== Select Reliable IDs ==================================>
    print('================> Select reliable images for the first stage re-training')

    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    select_reliable(checkpoints, dataloader, args)

    # <============================== Pseudo label reliable images =============================>
    print('================> Pseudo labeling reliable images')

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    label(model, dataloader, args)

    # <=============================== The 1st stage re-training ===============================>
    print('================> The 1st stage re-training on labeled and reliable unlabeled images')

    MODE = 'semi_train'

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                           args.labeled_id_path, cur_unlabeled_id_path, args.pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    model, optimizer = init_basic_elems(args)

    train(model, trainloader, valloader, criterion, optimizer, args)

    # <============================= Pseudo label unreliable images ============================>
    print('================> Pseudo labeling unreliable images')

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'unreliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    label(model, dataloader, args)

    # <=============================== The 2st stage re-training ===============================>
    print('================> The 2st stage re-training on labeled and all unlabeled images')

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                           args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    model, optimizer = init_basic_elems(args)

    train(model, trainloader, valloader, criterion, optimizer, args)


def init_basic_elems(args):
    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model = model_zoo[args.model](args.backbone, len(trainset.CLASSES))
    print('\nParams: %.1fM' % count_params(model))

    head_lr_multiple = 10.0
    if args.model == 'deeplabv2':
        assert args.backbone == 'resnet101'
        model.load_state_dict(torch.load('data/models/deeplabv2_resnet101_coco_pretrained.pth'))
        head_lr_multiple = 1.0

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
                     {'params': [param for name, param in model.named_parameters()
                                 if 'backbone' not in name],
                      'lr': args.lr * head_lr_multiple}],
                    lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model = DataParallel(model).cuda()

    return model, optimizer


def train(model, trainloader, valloader, criterion, optimizer, args):
    iters = 0
    total_iters = len(trainloader) * args.epochs

    previous_best = 0.0

    if MODE == 'train':
        checkpoints = []

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
            optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0

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
                os.remove(os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
            previous_best = mIOU
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))

        if MODE == 'train' and epoch == args.epochs // 3 or epoch == args.epochs * 2 // 2:
            checkpoints.append(deepcopy(model))

    if MODE == 'train':
        return checkpoints


def select_reliable(models, dataloader, args):
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

            mIOU = []
            for i in range(len(preds) - 1):
                metric = meanIOU(num_classes=num_classes)
                metric.add_batch(preds[i], preds[-1])
                mIOU.append(metric.evaluate()[-1])

            reliability = sum(mIOU) / len(mIOU)
            id_to_reliability.append((id[0], reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[:len(id_to_reliability) // 2]:
            f.write(elem[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2:]:
            f.write(elem[0] + '\n')


def label(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)

    metric = meanIOU(num_classes=len(dataloader.dataset.CLASSES))
    cmap = color_map(args.dataset)

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()
            pred = model(img, args.tta)
            pred = torch.argmax(pred, dim=1).cpu()

            metric.add_batch(pred.numpy(), mask.numpy())
            mIOU = metric.evaluate()[-1]

            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred.putpalette(cmap)

            pred.save('%s/%s.png' % (args.pseudo_mask_path, id[0].split(' ')[1]))

            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))


if __name__ == '__main__':
    args = parse_args()

    if args.epochs is None:
        args.epochs = {'pascal': 80, 'cityscapes': 240, 'coco': 30}[args.dataset]
    if args.lr is None:
        args.lr = {'pascal': 0.001, 'cityscapes': 0.004, 'coco': 0.004}[args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'pascal': 321, 'cityscapes': 721, 'coco': 321}[args.dataset]

    print(args)

    main(args)
