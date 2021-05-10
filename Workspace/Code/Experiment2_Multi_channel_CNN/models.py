# -*- coding: utf-8 -*-
"""
@Time    : 2019/5/27 19:24
@File    : models.py
@Software: PyCharm
Introduction: Models for diagnosing the heart disease by the ECG.
"""
#%% Import Packages
import numpy as np
import torch
import torch.nn as nn
#%% Functions
class MC_CNN(nn.Module):
    def __init__(self):
        super(MC_CNN, self).__init__()
        self.raw1 = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=(30, 1), stride=1, padding=(1, 0), bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout(0.8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.raw2 = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=(10, 1), stride=1, padding=(1, 0), bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout(0.8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 1))
        # convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(50, 1),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (30, 1), 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (10, 1), 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, (10, 1), 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, (5, 1), 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 1), 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.network = nn.Sequential(
            nn.Linear(512 * 2, 100),
            nn.Dropout(0.5),
            nn.Linear(100, 60),
            nn.Dropout(0.5),
            nn.Linear(60, 9),
        )

    def forward(self, x):
        # input（Batch_size * 12 * 4000 * 1）
        x1 = self.raw1(x)
        x2 = self.raw2(x)
        x = torch.cat((x1, x2), dim=2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x1= self.conv4(x)
        x =self.conv5(x1)
        x = torch.cat((x, x1), dim=2)
        x =self.conv6(x)
        x = nn.functional.avg_pool2d(x, 3)
        x = x.view(x.size(0), -1)
        output = self.network(x)
        return output
#%% Main Function
if __name__ == '__main__':
    x = np.random.randn(10, 12, 3000, 1)
    x = torch.Tensor(x)
    mccnn = MC_CNN()
    y = mccnn(x)
