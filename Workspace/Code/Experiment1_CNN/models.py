 # -*- coding: utf-8 -*-
"""
@Time    : 2019/5/25 21:12
@File    : models.py
@Software: PyCharm
Introduction: Models for diagnosing the heart disease by the ECG.
"""
#%% Import Packages
import torch
import torch.nn as nn
import numpy as np
#%% Functions
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolution layer
        self.conv1 = nn.Sequential(
            # convolution
            nn.Conv2d(
                in_channels=12,          # input channel
                out_channels=32,        # output channel
                kernel_size=(3,1),    # kernel size
                stride=1,               # length of kernel step
                padding=0,              # zero_padding
            ),
            # Rectification (by ReLU function)
            nn.ReLU(),
            # Pooling (by Max pooling)
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        # Second convolution layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 1), 1, 0),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 1), 1, 0),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 1), 1, 0),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 1), 1, 0),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 1), 1, 0),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 1), 1, 0),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.network = nn.Sequential(
            nn.Linear(22528, 100),  # Linear layer
            nn.Dropout(0.5),      # Dropout layer
            nn.Linear(100, 60),
            nn.Dropout(0.5),
            nn.Linear(60, 9),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(x.size(0), -1)
        output = self.network(x)
        return output

#%% Main Function
if __name__ == '__main__':
     x = np.random.randn(10, 12, 4000, 1)
     x = torch.Tensor(x)
     cnn = CNN()
     y = cnn(x)