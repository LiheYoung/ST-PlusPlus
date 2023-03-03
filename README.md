# ST++

This is the official PyTorch implementation of our CVPR 2022 paper:

> [**ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation**](https://arxiv.org/abs/2106.05095)       
> Lihe Yang, Wei Zhuo, Lei Qi, Yinghuan Shi, Yang Gao        
> *In Conference on Computer Vision and Pattern Recognition (CVPR), 2022*

We have another simple yet stronger end-to-end framework **UniMatch** accepted by CVPR 2023:

> **[Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2208.09910)** [[Code](https://github.com/LiheYoung/UniMatch)]</br>
> Lihe Yang, Lei Qi, Litong Feng, Wayne Zhang, Yinghuan Shi</br>
> *In Conference on Computer Vision and Pattern Recognition (CVPR), 2023*

## Getting Started

### Data Preparation

#### Pre-trained Model

[ResNet-50](https://download.pytorch.org/models/resnet50-0676ba61.pth) | [ResNet-101](https://download.pytorch.org/models/resnet101-63fe2227.pth) | [DeepLabv2-ResNet-101](https://drive.google.com/file/d/14be0R1544P5hBmpmtr8q5KeRAvGunc6i/view?usp=sharing)

#### Dataset

[Pascal JPEGImages](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | [Pascal SegmentationClass](https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing) | [Cityscapes leftImg8bit](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [Cityscapes gtFine](https://drive.google.com/file/d/1E_27g9tuHm6baBqcA7jct_jqcGA89QPm/view?usp=sharing) 

#### File Organization

```
├── ./pretrained
    ├── resnet50.pth
    ├── resnet101.pth
    └── deeplabv2_resnet101_coco_pretrained.pth
    
├── [Your Pascal Path]
    ├── JPEGImages
    └── SegmentationClass
    
├── [Your Cityscapes Path]
    ├── leftImg8bit
    └── gtFine
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

## Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{st++,
  title={ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation},
  author={Yang, Lihe and Zhuo, Wei and Qi, Lei and Shi, Yinghuan and Gao, Yang},
  booktitle={CVPR},
  year={2022}
}

@inproceedings{unimatch,
  title={Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation},
  author={Yang, Lihe and Qi, Lei and Feng, Litong and Zhang, Wayne and Shi, Yinghuan},
  booktitle={CVPR},
  year={2023}
}
```
