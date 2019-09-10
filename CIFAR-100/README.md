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
python train_with_distillation.py \
--setting e \
--epochs 200 \
--batch_size 64 \
--lr 0.1 \
--momentum 0.9 \
--weight_decay 5e-4
```

### Experimental results

| Setup |   Teacher   |   Student   | Original | Proposed | Improvement |
|:-----:|:-----------:|:-----------:|:--------:|:--------:|:-----------:|
|  (a)  |   WRN 28-4  |   WRN 16-4  |  22.72%  |  20.89%  |    1.83%    |
|  (b)  |   WRN 28-4  |   WRN 28-2  |  24.88%  |  21.98%  |    2.90%    |
|  (c)  |   WRN 28-4  |   WRN 16-2  |  27.32%  |  24.08%  |    3.24%    |
|  (d)  |   WRN 28-4  |  ResNet 56  |  27.68%  |  24.44%  |    3.24%    |
|  (f)  | Pyramid-200 |   WRN 28-4  |  21.09%  |  17.80%  |    3.29%    |
|  (g)  | Pyramid-200 | Pyramid-110 |  22.58%  |  18.89%  |    3.69%    |
