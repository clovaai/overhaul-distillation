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
python train_with_distillation.py \
--data_path your/path/to/ImageNet \
--net_type resnet \
--epochs 100 \
--lr 0.1 \
--batch_size 256
```

- (b) : ResNet50 to MobileNet
```
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
|            | Proposed |    __21.65__   |    __5.83__    |

- MobileNet

|  Network  |  Method  |  Top1 |  Top5 |
|:---------:|:--------:|:-----:|:-----:|
| ResNet 50 |  Teacher | 23.84 |  7.14 |
| Mobilenet | Original | 31.13 | 11.24 |
|           | Proposed | __28.75__ |  __9.66__ |
