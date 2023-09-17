# Feature-Suppressed Contrast for Self-Supervised Food Pre-training
This is a PyTorch implementation of the ACM MM 2023 paper. **[[ArXiv](https://arxiv.org/abs/2308.03272)]**

## Introduction

![Framework](framework.pdf)

Most previous approaches for analyzing food images have relied on extensively annotated datasets, resulting in significant human labeling expenses due to the varied and intricate nature of such images. Inspired by the effectiveness of contrastive self-supervised methods in utilizing unlabelled data, weiqing explore leveraging these techniques on unlabelled food images. In contrastive self-supervised methods, two views are randomly generated from an image by data augmentations. However, regarding food images, the two views tend to contain similar informative contents, causing large mutual information, which impedes the efficacy of contrastive self-supervised learning. To address this problem, we propose Feature Suppressed Contrast (FeaSC) to reduce mutual information between views. As the similar contents of the two views are salient or highly responsive in the feature map, the proposed FeaSC uses a response-aware scheme to localize salient features in an unsupervised manner. By suppressing some salient features in one view while leaving another contrast view unchanged, the mutual information between the two views is reduced, thereby enhancing the effectiveness of contrast learning for self-supervised food pre-training. As a plug-and-play module, the proposed method consistently improves BYOL and SimSiam by 1.70\% ∼ 6.69\% classification accuracy on four publicly available food recognition datasets. Superior results have also been achieved on downstream segmentation tasks, demonstrating the effectiveness of the proposed method.


## Installation

We perform self-supervised pre-training using the proposed method on [Food2K dataset](http://123.57.42.89/FoodProject.html). Please prepare the dataset accordingly. This codebase is tested in the following environments and should also be compatible for later versions.

* Ubuntu 20.04
* Python 3.7
* [PyTorch](https://pytorch.org) 1.7.1
* [Torchvision](https://pytorch.org)  0.8.2 
* cudatoolkit 10.2


## Pretraining

To do self-supervised pre-training of a ResNet-50 model for 100 epochs using an 4-gpu machine, simply run:

```bash
$ cd BYOL_FSC
$ bash pretrain.sh
```

```

## Linear Probing

To do

```

## Models and Logs

To do

## Acknowledgement

This repo is mainly based on [SimSiam](https://github.com/facebookresearch/simsiam). Many thanks to their wonderful work!


## Citation

```
@inproceedings{
liu2023feature,
title={Feature-Suppressed Contrast for Self-Supervised Food Pre-training},
author={Liu, Xinda and Zhu, Yaohui and Liu, Linhu and Tian, Jiang and Wang, Lili},
booktitle={31st ACM International Conference on Multimedia (ACM MM 2023)},
year={2023},
}
```

If you have any questions or suggestions about this work, please feel free to contact me.



