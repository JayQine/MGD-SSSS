# MGD-SSSS
This code is the official implementation of our paper [Multi-Granularity Distillation Scheme Towards Lightweight Semi-Supervised Semantic Segmentation](https://arxiv.org/pdf/2208.10169.pdf), ECCV2022.

## Abstract
Albeit with varying degrees of progress in the field of Semi-Supervised Semantic Segmentation, most of its recent successes are involved in unwieldy models and the lightweight solution is still not yet explored. We find that existing knowledge distillation techniques pay more attention to pixel-level concepts from labeled data, which fails to take more informative cues within unlabeled data into account. Consequently, we offer the first attempt to provide lightweight SSSS models via a novel multi-granularity distillation (MGD) scheme, where multi-granularity is captured from three aspects: i) complementary teacher structure; ii) labeled-unlabeled data cooperative distillation; iii) hierarchical and multi-levels loss setting. Specifically, MGD is formulated as a labeled-unlabeled data cooperative distillation scheme, which helps to take full advantage of diverse data characteristics that are essential in the semi-supervised setting. Image-level semantic-sensitive loss, region-level content-aware loss, and pixel-level consistency loss are set up to enrich hierarchical distillation abstraction via structurally complementary teachers. Experimental results on PASCAL VOC2012 and Cityscapes reveal that MGD can outperform the competitive approaches by a large margin under diverse partition protocols. For example, the performance of ResNet-18 and MobileNet-v2 backbone is boosted by 11.5% and 4.6% respectively under 1/16 partition protocol on Cityscapes. Although the FLOPs of the model backbone is compressed by 3.4-5.3× (ResNet-18) and 38.7-59.6× (MobileNetv2), the model manages to achieve satisfactory segmentation results.

## Method
<div>
	<img src="img/method.png" width="800" height="800">
</div>

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
