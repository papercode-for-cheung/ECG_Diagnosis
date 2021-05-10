# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/1 19:58
@Author  : QuYue
@File    : model.py
@Software: PyCharm
Introduction: Design the models using pytorch.
"""
#%% Import Packages
import numpy as np
import torch
import torch.nn as nn
from transformer_block import TRANSFORMER
#%% Functions
class Diagnosis(nn.Module):
    def __init__(self):
        super(Diagnosis, self).__init__()
        # 第一个卷积层（包括了卷积、整流、池化）
        self.conv1 = nn.Sequential(
            # 卷积
            nn.Conv2d(
                in_channels=12,  # 输入通道
                out_channels=32,  # 输出通道
                kernel_size=(30, 1),  # 卷积核的长度
                stride=1,  # 步长 （卷积核移动的步长）
                padding=0,  # 补0 zero_padding （对图片的边缘加上一圈0，防止越卷积图片越小）
            ),
            # 整流 使用ReLU函数进行整流
            nn.ReLU(),
            # 池化 使用最大池化
            nn.MaxPool2d(kernel_size=(3, 1))  # 最大池化 在3*1的范围内取最大
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (10, 1), 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, (10, 1), 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, (10, 1), 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1))
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, (5, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, (2, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((1, 1))
        )
        self.network = nn.Sequential(
            nn.Linear(512, 100),  # 卷积层后加一个普通的神经网络
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(100, 60),  # 卷积层后加一个普通的神经网络
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(60, 2),  # 卷积层后加一个普通的神经网络
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        print(x.shape)
        x = x.view(x.size(0), -1)  # 将数据展平 (Batch_size * 28160)
        output = self.network(x)
        return output
#%%
class Diagnosis2(nn.Module):
    def __init__(self):
        super(Diagnosis2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(12, 32, (30, 1), 1, 0),
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
            nn.Conv2d(64, 64, (5, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, (2, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((1, 1))
        )
        self.bilstm = nn.LSTM(128, 256, batch_first=True, bidirectional=True)
        self.network = nn.Sequential(
            nn.Linear(128 * 1, 100),  # 卷积层后加一个普通的神经网络
            nn.Dropout(0.4),
            nn.Linear(100, 60),  # 卷积层后加一个普通的神经网络
            nn.Dropout(0.4),
            nn.Linear(60, 2),  # 卷积层后加一个普通的神经网络
        )
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.transform = TRANSFORMER()
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.reshape(x.size(0), 128, -1)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.transform(x)
        # print(x.shape)
        x = self.dropout(x)
        # x = self.batchnorm(x)
        x = x[:,-1,:]
        x = self.network(x)
        return x
#%%
class Diagnosis3(nn.Module):
    def __init__(self):
        super(Diagnosis3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(12, 32, (30, 1), 1, 0),
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
            nn.Conv2d(64, 64, (5, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, (2, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((1, 1))
        )
        self.bilstm = nn.LSTM(128, 256, batch_first=True, bidirectional=True)
        self.network = nn.Sequential(
            nn.Linear(512 * 1, 100),  # 卷积层后加一个普通的神经网络
            nn.Dropout(0.4),
            nn.Linear(100, 60),  # 卷积层后加一个普通的神经网络
            nn.Dropout(0.4),
            nn.Linear(60, 2),  # 卷积层后加一个普通的神经网络
        )
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.reshape(x.size(0), 128, -1)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = self.relu(x)
        r_out, (h_n, c_n) = self.bilstm(x, None)
        x = r_out[:, -1, :]
        x = self.dropout(x)
        x = self.batchnorm(x)
        x = self.network(x)
        return x



#%% Main Function
if __name__ == '__main__':
    diagnosis = Diagnosis2()
    x = np.random.randn(10, 12, 5000, 1)
    x = torch.Tensor(x)
    x = diagnosis(x)





