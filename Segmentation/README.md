## Segmentation - Pascal VOC

Our segmentation code is based on [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception).

### Additional requirements

- tqdm
- matplotlib 
- pillow

### Settings

|   Teacher  |  Student  | Teacher size | Student size | Size ratio |
|:----------:|:---------:|:------------:|:------------:|:----------:|
| ResNet 101 | ResNet 18 |    59.3M    |    16.6    |   28.0%   |
| ResNet 101 | MobileNetV2 |    59.3M    |     5.8M    |   9.8%   |


### Teacher models
Download following pre-trained teacher network and put it into ```./Segmentation/pretrained``` directory
- [ResNet101-DeepLabV3+](https://drive.google.com/open?id=1Pz2OT5KoSNvU5rc3w5d2R8_0OBkKSkLR)

We used pre-trained model in [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception) for teacher network.

### Training

- First, move to segmentation folder : ```cd Segmentation```
- Next, configure your dataset path on ```Segmentation/mypath.py```

- Without distillation
  - ResNet 18
  ```shell script
  CUDA_VISIBLE_DEVICES=0,1 python train.py --backbone resnet18 --gpu-ids 0,1 --dataset pascal --use-sbd --nesterov
  ```
  
  - MobileNetV2
  ```shell script
  CUDA_VISIBLE_DEVICES=0,1 python train.py --backbone mobilenet --gpu-ids 0,1 --dataset pascal --use-sbd --nesterov
  ````

- Distillation
  - ResNet 18
  ```shell script
  CUDA_VISIBLE_DEVICES=0,1 python train_with_distillation.py --backbone resnet18 --gpu-ids 0,1 --dataset pascal --use-sbd --nesterov
  ```
  
  -MobileNetV2
  ```shell script
  CUDA_VISIBLE_DEVICES=0,1 python train_with_distillation.py --backbone mobilenet --gpu-ids 0,1 --dataset pascal --use-sbd --nesterov
  ```

### Experimental results

This numbers are based validation performance of our code.

- ResNet 18

|   Network  |  Method  | mIOU |
|:----------:|:--------:|:----------:|
| ResNet 101 |  Teacher |    77.89   |
| ResNet 18 | Original |    72.07   |
| ResNet 18 | Proposed |    __73.98__   |

- MobileNetV2

|  Network  |  Method  |  mIOU |
|:---------:|:--------:|:-----:|
| ResNet 101 |  Teacher | 77.89 |
| MobileNetV2 | Original | 68.46 |
| MobileNetV2 | Proposed | __71.19__ |


In the paper, we reported performance on the **test** set, but our code measures the performance on the **val** set.
Therefore, the performance on code is not same as the paper.
If you want accurate measure, please measure performance on **test** set with [Pascal VOC evaluation server](http://host.robots.ox.ac.uk/pascal/VOC/).
