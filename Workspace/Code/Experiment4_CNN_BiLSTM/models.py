# -*- coding: utf-8 -*-
"""
@Time    : 2019/5/28 1:17
@File    : models.py
@Software: PyCharm
Introduction: Models for diagnosing the heart disease by the ECG.
"""
#%% Import Packages
import numpy as np
import torch
import torch.nn as nn
#%% Functions
class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=32,
                kernel_size=(30, 1),
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (10, 1), 1, 0),
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (10, 1), 1, 0),
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, (5, 1), 1, 0),
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 1), 1, 0),
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 1), 1, 0),
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, (2, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((1, 1))
        )
        self.network1 = nn.Sequential(
            nn.Linear(256, 200),  # 卷积层后加一个普通的神经网络
            # nn.BatchNorm1d(256),
            nn.Dropout(0.7),
            nn.ReLU(),
            nn.Linear(200, 128),  # 卷积层后加一个普通的神经网络
            # nn.BatchNorm1d(100),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(128, 256, num_layers=1, batch_first=True, bidirectional=True)
        self.networks2 = nn.Sequential(
            nn.Linear(256 * 2, 100),
            nn.Dropout(0.7),
            nn.Linear(100, 9),
        )
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # x = self.conv7(x)
        x = x.reshape(x.shape[0], -1)  # win_num * feature
        x = self.network1(x)
        x = x.unsqueeze(0)
        x = self.dropout(x)
        x, (h_n, c_n) = self.bilstm(self.relu(x))
        x = h_n[:2, :, :]
        x = h_n.reshape(1, -1)
        x = self.networks2(x)
        return x

#%% Main Function
if __name__ == '__main__':
    x = np.random.randn(3, 12, 3000, 1)
    x = torch.Tensor(x)
    cnn_bilstm = CNN_BiLSTM()
    y = cnn_bilstm(x)
    print(y.shape)
