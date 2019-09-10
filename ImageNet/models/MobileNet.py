'''MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

    def get_bn_before_relu(self):
        bn1 = self.model[3][-2]
        bn2 = self.model[5][-2]
        bn3 = self.model[11][-2]
        bn4 = self.model[13][-2]

        return [bn1, bn2, bn3, bn4]

    def get_channel_num(self):

        return [128, 256, 512, 1024]

    def extract_feature(self, x, preReLU=False):

        feat1 = self.model[3][:-1](self.model[0:3](x))
        feat2 = self.model[5][:-1](self.model[4:5](F.relu(feat1)))
        feat3 = self.model[11][:-1](self.model[6:11](F.relu(feat2)))
        feat4 = self.model[13][:-1](self.model[12:13](F.relu(feat3)))

        out = self.model[14](F.relu(feat4))
        out = out.view(-1, 1024)
        out = self.fc(out)

        if not preReLU:
            feat1 = F.relu(feat1)
            feat2 = F.relu(feat2)
            feat3 = F.relu(feat3)
            feat4 = F.relu(feat4)

        return [feat1, feat2, feat3, feat4], out
