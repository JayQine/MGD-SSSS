# MGD-SSSS
This code is the official implementation of our paper Multi-Granularity Distillation Scheme Towards Lightweight Semi-Supervised Semantic Segmentation, ECCV2022.


## Installation

### Environment
* python=3.7
* torch=1.8.1
* torchvision=0.9.1
* timm=0.5.4
* apex=0.1

Other dependency
```
pip install -r requirement.txt
```


## Data Preparation
Download the datasets ([VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html), [Cityscapes](https://www.cityscapes-dataset.com/)) to follow folders. 

```
DATA/
|-- city
|   |-- config_new
|   |-- images
|   |   |-- train
|   |   |-- val
|   |   |-- test
|   |-- segmentation
|   |   |-- train
|   |   |-- val
|   |   |-- test
|-- pascal_voc
|   |-- subset_train_aug
|   |-- train_aug
|   |   |-- image
|   |   |-- label
|   |-- val
|   |   |-- image
|   |   |-- label
|   |-- ...
|-- model-weight
```

## Getting Start

### Train & Test
Please first download the pretrained [models](https://drive.google.com/drive/folders/1dBzx9Ows9YqNVwboUNMFBWLSl50Xwt28?usp=sharing) to ``DATA/model-weight``.

To train the lightweight model ResNet18 on 1/8 partition of VOC2012, you need to specify some variables in `script.sh`, such as the path of data dir, the path of snapshot, and the output dir, etc. 

```shell
$ cd ./exp.voc/voc8.resnet18_deeplabv3plus
$ bash script.sh
```

## Citation
If you find this work useful in your method, you can cite the paper as below:

```shell
@inproceedings{qin2022multi,
  title={Multi-granularity distillation scheme towards lightweight semi-supervised semantic segmentation},
  author={Qin, Jie and Wu, Jie and Li, Ming and Xiao, Xuefeng and Zheng, Min and Wang, Xingang},
  booktitle={European Conference on Computer Vision},
  pages={481--498},
  year={2022},
  organization={Springer}
}
```
