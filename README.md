# ST++

**Implementation for 'ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation'**

## Getting Started

### Data Preparation

#### Pre-trained Model: [ResNet-50](https://download.pytorch.org/models/resnet50-0676ba61.pth) | [ResNet-101](https://download.pytorch.org/models/resnet101-63fe2227.pth) | [DeepLabv2-ResNet-101](https://drive.google.com/file/d/14be0R1544P5hBmpmtr8q5KeRAvGunc6i/view?usp=sharing)

#### Dataset: [Pascal](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | [Masks of Pascal (Augmented)](https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing) | [Cityscapes](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [Masks of Cityscapes (Class Mapped)](https://drive.google.com/file/d/1E_27g9tuHm6baBqcA7jct_jqcGA89QPm/view?usp=sharing) 

#### File Organization

```
├── ./pretrained
    ├── resnet-50.pth
    ├── resnet-101.pth
    └── deeplabv2_resnet101_coco_pretrained.pth
    
├── Pascal
    ├── JPEGImages
    └── SegmentationClass
    
├── Cityscapes
    ├── leftImg8bit
    └── gtFine
```



### Running Scripts


#### Training


#### Pseudo Labeling


#### Re-training
