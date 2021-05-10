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
class CNN_BiLSTM_BiLSTM(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM_BiLSTM, self).__init__()
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
        self.bilstm1 = nn.LSTM(128, 256, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.network1 = nn.Sequential(
            nn.Linear(512 * 1, 100),  # 卷积层后加一个普通的神经网络
            nn.Dropout(0.4),
            nn.Linear(100, 60),  # 卷积层后加一个普通的神经网络
            nn.Dropout(0.4),
            nn.Linear(60, 16),  # 卷积层后加一个普通的神经网络
        )
        self.dropout2 = nn.Dropout(0.5)
        self.bilstm2 = nn.LSTM(16, 64, batch_first=True, bidirectional=True)
        self.relu2 = nn.ReLU()
        self.network2 = nn.Sequential(
                nn.Linear(128, 64),
                nn.Dropout(0.5),
                nn.Linear(64, 9)
         )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.reshape(x.size(0), 128, -1)
        x = x.permute(0, 2, 1)
        x = self.dropout1(x)
        x = self.relu1(x)
        r_out, (h_n, c_n) = self.bilstm1(x)
        x = h_n[:2, :, :]
        x = x.permute(1, 0, 2)
        x = x.reshape(x.size(0), -1)
        x = self.dropout2(x)
        x = self.batchnorm1(x)
        x = self.network1(x)
        x = x.unsqueeze(0)
        output, (h_n, c_n) = self.bilstm2(self.relu2(x))
        x = h_n[:2, :, :]
        x = x.reshape(1, -1)
        x = self.network2(self.relu2(x))
        return x
#%% Main Function
if __name__ == '__main__':
    x = np.random.randn(3, 12, 3000, 1)
    x = torch.Tensor(x)
    cnn_bilstm2 = CNN_BiLSTM_BiLSTM()
    y = cnn_bilstm2(x)
    print(y.shape)
