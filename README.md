# FRS Code

Implementation of our NeurIPS2021 paper [Distilling Object Detectors with Feature Richness](https://arxiv.org/pdf/2111.00674.pdf)

## Abstract

In recent years, large-scale deep models have achieved great success, but the huge computational complexity and massive storage requirements make it a great challenge to deploy them in resource-limited devices. As a model compression and acceleration method, knowledge distillation effectively improves the performance of small models by transferring the dark knowledge from the teacher detector. However, most of the existing distillation-based detection methods mainly imitating features near bounding boxes, which suffer from two limitations. First, they ignore the beneficial features outside the bounding boxes. Second, these methods imitate some features which are mistakenly regarded as the background by the teacher detector. To address the above issues, we propose a novel Feature-Richness Score (FRS) method to choose important features that improve generalized detectability during distilling. The proposed method effectively retrieves the important features outside the bounding boxes and removes the detrimental features within the bounding boxes. Extensive experiments show that our methods achieve excellent performance on both anchor-based and anchor-free detectors. For example, RetinaNet with ResNet-50 achieves 39.7% in mAP on the COCO2017 dataset, which even surpasses the ResNet-101 based teacher detector 38.9% by 0.8%.

## Model

As shown in the Figure, we use the aggregated classification score to represent the feature richness and then perform distillation on both FPN layers and the classification head.

![Model](./picture/model.pdf)

## Install

Please refer to the [install.md](https://github.com/duzhixing/FRS/blob/master/install.md)

## Prerequisites

`python 3.7` `mmdetection` `PyTorch`



