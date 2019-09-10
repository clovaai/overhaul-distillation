# A Comprehensive Overhaul of Feature Distillation
**Accepted at ICCV 2019**

Official PyTorch implementation of "A Comprehensive Overhaul of Feature Distillation" | [paper](https://arxiv.org/abs/1904.01866) | [project page](https://sites.google.com/view/byeongho-heo/overhaul) | [blog](https://clova-ai.blog/2019/08/22/a-comprehensive-overhaul-of-feature-distillation-iccv-2019)

Byeongho Heo, Jeesoo Kim, Sangdoo Yun, Hyojin Park, Nojun Kwak, Jin Young Choi

Clova AI Research, NAVER Corp. \
Seoul National University

## Requirements
- Python3
- PyTorch (> 0.4.1)
- torchvision
- numpy
- scipy

## Updates
***10 Sep 2019*** Initial upload

## CIFAR-100

### Settings
We provide the code of the experimental settings specified in the paper.

| Setup | Compression type |   Teacher   |   Student   | Teacher size | Student size | Size ratio |
|:-----:|:----------------:|:-----------:|:-----------:|:------------:|:------------:|:----------:|
|  (a)  |       Depth      |   WRN 28-4  |   WRN 16-4  |     5.87M    |     2.77M    |    47.2%   |
|  (b)  |      Channel     |   WRN 28-4  |   WRN 28-2  |     5.87M    |     1.47M    |    25.0%   |
|  (c)  |  Depth & channel |   WRN 28-4  |   WRN 16-2  |     5.87M    |     0.70M    |    11.9%   |
|  (d)  |   Architecture   |   WRN 28-4  |  ResNet 56  |     5.87M    |     0.86M    |    14.7%   |
|  (e)  |   Architecture   | Pyramid-200 |   WRN 28-4  |    26.84M    |     5.87M    |    21.9%   |
|  (f)  |   Architecture   | Pyramid-200 | Pyramid-110 |    26.84M    |     3.91M    |    14.6%   |

### Teacher models
Download following pre-trained teacher network and put them into ```./data``` directory
- [Wide Residual Network 28-4](https://drive.google.com/open?id=1Quxgs5teXVXwD3jBdkk-WeNLNpxbiZXN)
- [PyramidNet-200(240)](https://drive.google.com/open?id=1_QgG81fNM3OvVIbMAxDPykKWuSIyKnmz)

### Training
Run ```CIFAR-100/train_with_distillation.py``` with setting alphabet (a - f)
```
cd CIFAR-100
python train_with_distillation.py \
--setting a \
--epochs 200 \
--batch_size 128 \
--lr 0.1 \
--momentum 0.9 \
--weight_decay 5e-4
```

For pyramid teacher (e, f), we used batch-size 64 to save gpu memory.
```
cd CIFAR-100
python train_with_distillation.py \
--setting e \
--epochs 200 \
--batch_size 64 \
--lr 0.1 \
--momentum 0.9 \
--weight_decay 5e-4
```

### Experimental results

Performance measure is classification error rate (%)


| Setup |   Teacher   |   Student   | Original | Proposed | Improvement |
|:-----:|:-----------:|:-----------:|:--------:|:--------:|:-----------:|
|  (a)  |   WRN 28-4  |   WRN 16-4  |  22.72%  |  20.89%  |    1.83%    |
|  (b)  |   WRN 28-4  |   WRN 28-2  |  24.88%  |  21.98%  |    2.90%    |
|  (c)  |   WRN 28-4  |   WRN 16-2  |  27.32%  |  24.08%  |    3.24%    |
|  (d)  |   WRN 28-4  |  ResNet 56  |  27.68%  |  24.44%  |    3.24%    |
|  (f)  | Pyramid-200 |   WRN 28-4  |  21.09%  |  17.80%  |    3.29%    |
|  (g)  | Pyramid-200 | Pyramid-110 |  22.58%  |  18.89%  |    3.69%    |

## ImageNet

### Settings

| Setup | Compression type |   Teacher  |  Student  | Teacher size | Student size | Size ratio |
|:-----:|:----------------:|:----------:|:---------:|:------------:|:------------:|:----------:|
|  (a)  |       Depth      | ResNet 152 | ResNet 50 |    60.19M    |    25.56M    |   42.47%   |
|  (b)  |   Architecture   |  ResNet 50 | MobileNet |    25.56M    |     4.23M    |   16.55%   |


In case of ImageNet, teacher model will be automatically downloaded from PyTorch sites.

### Training

- (a) : ResNet152 to ResNet50
```
cd ImageNet
python train_with_distillation.py \
--data_path your/path/to/ImageNet \
--net_type resnet \
--epochs 100 \
--lr 0.1 \
--batch_size 256
```

- (b) : ResNet50 to MobileNet
```
cd ImageNet
python train_with_distillation.py \
--data_path your/path/to/ImageNet \
--net_type mobilenet \
--epochs 100 \
--lr 0.1 \
--batch_size 256
```

### Experimental results

- ResNet 50

|   Network  |  Method  | Top1-error | Top5-error |
|:----------:|:--------:|:----------:|:----------:|
| ResNet 152 |  Teacher |    21.69   |    5.95    |
|  ResNet 50 | Original |    23.72   |    6.97    |
|  ResNet 50 | Proposed |    __21.65__   |    __5.83__    |

- MobileNet

|  Network  |  Method  |  Top1-error |  Top5-error |
|:---------:|:--------:|:-----:|:-----:|
| ResNet 50 |  Teacher | 23.84 |  7.14 |
| Mobilenet | Original | 31.13 | 11.24 |
| Mobilenet | Proposed | __28.75__ |  __9.66__ |

## Citation

```
@inproceedings{heo2019overhaul,
  title={A Comprehensive Overhaul of Feature Distillation},
  author={Heo, Byeongho and Kim, Jeesoo and Yun, Sangdoo and Park, Hyojin and Kwak, Nojun and Choi, Jin Young},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

## License

```
Copyright (c) 2019-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
