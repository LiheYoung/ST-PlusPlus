## Towards Making Self-Training Work Better for Semi-Supervised Semantic Segmentation

This is the official PyTorch implementation of our paper:

Towards Making Self-Training Work Better for Semi-Supervised Semantic Segmentation.

## Getting Started

### Data Preparation

#### Pre-trained Model

[ResNet-50](https://download.pytorch.org/models/resnet50-0676ba61.pth) | [ResNet-101](https://download.pytorch.org/models/resnet101-63fe2227.pth) | [DeepLabv2-ResNet-101](https://drive.google.com/file/d/14be0R1544P5hBmpmtr8q5KeRAvGunc6i/view?usp=sharing)

#### Dataset

[Pascal](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | [Augmented Masks](https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing) | [Cityscapes](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [Class Mapped Masks](https://drive.google.com/file/d/1E_27g9tuHm6baBqcA7jct_jqcGA89QPm/view?usp=sharing) 

#### File Organization

```
├── ./pretrained
    ├── resnet50.pth
    ├── resnet101.pth
    └── deeplabv2_resnet101_coco_pretrained.pth
    
├── [Your Pascal Path]
    ├── JPEGImages
    └── SegmentationClass    # replace the official folder with above augmented masks 
    
├── [Your Cityscapes Path]
    ├── gtFine               # replace the official folder with above class mapped masks 
    └── leftImg8bit
```


### Training and Testing

```
export semi_setting='pascal/1_8/split_0'

CUDA_VISIBLE_DEVICES=0,1 python -W ignore main.py \
  --dataset pascal --data-root [Your Pascal Path] \
  --batch-size 16 --backbone resnet50 --model deeplabv3plus \
  --labeled-id-path dataset/splits/$semi_setting/labeled.txt \
  --unlabeled-id-path dataset/splits/$semi_setting/unlabeled.txt \
  --pseudo-mask-path outdir/pseudo_masks/$semi_setting \
  --save-path outdir/models/$semi_setting
```
This script is for our ST framework. To run ST++, add ```--plus --reliable-id-path outdir/reliable_ids/$semi_setting```.


## Acknowledgement

The DeepLabv2 MS COCO pre-trained model is borrowed and converted from **AdvSemiSeg**.
The image partitions are borrowed from **Context-Aware-Consistency** and **PseudoSeg**. 
Part of the training hyper-parameters and network structures are adapted from **PyTorch-Encoding**. The strong data augmentations are borrowed from **MoCo v2** and **PseudoSeg**.
 
+ AdvSemiSeg: [https://github.com/hfslyc/AdvSemiSeg](https://github.com/hfslyc/AdvSemiSeg).
+ Context-Aware-Consistency: [https://github.com/dvlab-research/Context-Aware-Consistency](https://github.com/dvlab-research/Context-Aware-Consistency).
+ PseudoSeg: [https://github.com/googleinterns/wss](https://github.com/googleinterns/wss).
+ PyTorch-Encoding: [https://github.com/zhanghang1989/PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding).
+ MoCo: [https://github.com/facebookresearch/moco](https://github.com/facebookresearch/moco).
+ OpenSelfSup: [https://github.com/open-mmlab/OpenSelfSup](https://github.com/open-mmlab/OpenSelfSup).

Thanks a lot for their great works!
