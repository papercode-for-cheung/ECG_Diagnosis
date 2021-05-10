# -*- coding: utf-8 -*-
"""
@Time    : 2019/5/27 20:21
@File    : models.py
@Software: PyCharm
Introduction: Models for diagnosing the heart disease by the ECG.
"""
#%% Import Packages
import numpy as np
import torch
import torch.nn as nn
#%% Functions
class ResidualBlock(nn.Module):
    # Residual Block
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 1), stride=stride, padding=(1, 0), bias=False),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False),
            nn.BatchNorm2d(num_features=out_channel)
        )
        self.right = shortcut  # sth like nn.Module
        # self.right = nn.Sequential(
        #     nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), padding=0,stride=stride, bias=True),
        #     nn.BatchNorm2d(num_features=out_channel))

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return nn.functional.relu(out)  # Relu(out + residual)

class ResNet34(nn.Module):
    # ResNet34: including 5 layers
    def __init__(self):
        super(ResNet34, self).__init__()
        '''
         {(input_height - kernel_size + 2*padding) / stride[0] }+1
         {(input_Weight - kernel_size + 2*padding) / stride[1] }+1
        '''
        # First convolution and pooling layer
        self.pre = nn.Sequential(
            # nn.Conv2d(12,12,(1,1),stride=1,padding=0,bias=False),
            # nn.BatchNorm2d(12),
            # nn.Dropout(0.7),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=(55, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout(0.8),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(3, 1))
        )
        # nn.Conv2d(32, 32, (1, 1), stride=1, padding=0, bias=False),
        # nn.BatchNorm2d(32),
        # nn.Dropout(0.7),
        # nn.ReLU(inplace=True))

        # 5 layers : which include 3、4、6、3 Residual Blocks
        self.layer1 = self.make_layer(32, 32, 3)
        self.layer2 = self.make_layer(32, 64, 4, stride=2)
        self.layer3 = self.make_layer(64, 128, 6, stride=2)
        self.layer4 = self.make_layer(128, 128, 3, stride=2)

        self.network = nn.Sequential(
            nn.Linear(15744 * 1, 1000),
            nn.Dropout(0.5),
            nn.Linear(1000, 100),
            nn.Dropout(0.5),
            nn.Linear(100, 9),
        )
        # self.fc = nn.Linear(in_features=512, out_features=num_target_classes)

    def make_layer(self, in_channel, out_channel, num_blocks, stride=1):
        # create layers (include a few of block)
        shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
            nn.BatchNorm2d(num_features=out_channel)
        )
        # kernel size=1
        layers = [ResidualBlock(in_channel, out_channel, stride, shortcut)]  # first block with shortcut
        for i in range(1, num_blocks):
            layers.append(ResidualBlock(out_channel, out_channel))  # other blocks without shortcut
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.layer3(x)
        x = self.layer4(x1)
        x = torch.cat((x1, x), dim=2)
        x = nn.functional.avg_pool2d(x, 3)
        x = x.view(x.size(0), -1)
        output = self.network(x)
        return output

#%% Main Function
if __name__ == '__main__':
    x = np.random.randn(10, 12, 3000, 1)
    x = torch.Tensor(x)
    resnet = ResNet34()
    y = resnet(x)