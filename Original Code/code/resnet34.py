# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 10:43:40 2019

@author: Aisunny
"""
'''
resnet
'''
#%%
import os
import time
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as t
import torch.nn.functional as f
from torch import nn
from torch import optim
import torch.autograd as autograd
import scipy.io as sio
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from sklearn.model_selection import train_test_split
from score_py3 import score
#%%
class ResidualBlock(nn.Module):
    """
    Residual Block
    """

    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3,1), stride=stride, padding=(1,0), bias=False),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3,1), stride=1, padding=(1,0), bias=False),
            nn.BatchNorm2d(num_features=out_channel)
        )
        self.right = shortcut  # sth like nn.Module
#        self.right = nn.Sequential(
#            nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), padding=0,stride=stride, bias=True),
#            nn.BatchNorm2d(num_features=out_channel)
#        )
    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return f.relu(out)  # 加上residual后，再ReLU
#%%
class ResNet34(nn.Module):
    """
    ResNet34
    包含5个layer
    """

    def __init__(self):
        super(ResNet34, self).__init__()
        # 最前面的卷积层和池化层
        '''
         {(input_height - kernel_size + 2*padding) / stride[0] }+1
         {(input_Weight - kernel_size + 2*padding) / stride[1] }+1
        '''
        self.pre = nn.Sequential(
            #
            # nn.Conv2d(12,12,(1,1),stride=1,padding=0,bias=False),
            # nn.BatchNorm2d(12),
            # nn.Dropout(0.7),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=(55,1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout(0.8),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(3,1))
                         )
            # nn.Conv2d(32, 32, (1, 1), stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(32),
            # nn.Dropout(0.7),
            # nn.ReLU(inplace=True))
        # batch channel width height
        # [none,32,1318,1]
        # 5个layer，分别包含3、4、6、3个Residual Block
        self.layer1 = self.make_layer(32, 32, 3)
        self.layer2 = self.make_layer(32, 64, 4, stride=2)
        self.layer3 = self.make_layer(64, 128, 6, stride=2)
        self.layer4 = self.make_layer(128, 128, 3, stride=2)

        self.network = nn.Sequential(
                nn.Linear(15744* 1, 1000), # 卷积层后加一个普通的神经网络
                nn.Dropout(0.5),
                nn.Linear(1000, 100), # 卷积层后加一个普通的神经网络
                nn.Dropout(0.5),
                nn.Linear(100, 9), # 卷积层后加一个普通的神经网络
        )
        # 最后的全连接层
#        self.fc = nn.Linear(in_features=512, out_features=num_target_classes)
    def make_layer(self, in_channel, out_channel, num_blocks, stride=1):
        """
        构建layer，包含多个block
        """
        shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1,1),stride=stride, bias=False),
            nn.BatchNorm2d(num_features=out_channel)
        )
        #
       #[1,out_channel ,lenght,1]形状不变的 
      #  [1,32,1318,1]
        # kernel size为1的卷积
        layers = [ResidualBlock(in_channel, out_channel, stride, shortcut)]  # 第一个block包含shortcut
        for i in range(1, num_blocks):
            layers.append(ResidualBlock(out_channel, out_channel))  # 之后的block不含shortcut
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x1= self.layer3(x)
        x = self.layer4(x1)
        # x = nn.MaxPool2d(kernel_size=(3,1))
        # x = self.pool()
        # x = nn.functional.avg_pool2d(x,3)
        # 合并layers的特征和 全连接层的特征
        # t
        x = t.cat((x1,x),dim=2)
        x = nn.functional.avg_pool2d(x, 3)
        x = x.view(x.size(0), -1)
        output= self.network(x)
        return output

#%% Set Hyper Parameter 超参数设置
class Parameter:
    # 超参数类
    def __init__(self):
        self.batch_size = 128 # Batch的大小
        self.epoch = 80# 迭代次数
        self.lr = 0.0001         # Learning rate 学习率，越大学习速度越快，越小学习速度越慢，但是精度可能会提高
        self.ifGPU = False      # 是否需要使用GPU（需要电脑里有NVIDIA的独立显卡并安装CUDA驱动才可以)
        self.trainratio = 0.9   # 训练集的比例
PARM = Parameter()
PARM.ifGPU =  True      # 开启GPU
PARM.datanum = 4000    # 提取数据量
PARM.len = 3000         # 数据截断长度
PATH = 'E:\\zhumin\\TrainingSet1\\TrainingSet1\\'
LABEL = 'REFERENCE.csv'
#%% Input Data
# 特征数据
count = 0
ECG_feature_data = []
for file in os.listdir(PATH):
    count +=1
    if count > PARM.datanum:
        break
    data = sio.loadmat(PATH+file)
    data = data['ECG']['data'][0][0]
    ECG_feature_data.append(data)
# 标签数据
ECG_label_data = pd.read_csv(PATH+LABEL)
ECG_label_data = ECG_label_data.iloc[0: PARM.datanum]
#%% Reprocess Data 数据预处理
X = np.zeros([PARM.datanum, 12, PARM.len])
for i in range(len(ECG_feature_data)):
    X[i,:] =  ECG_feature_data[i][:, :PARM.len]
X = X[:,:,:,np.newaxis]
Y = np.array(list(ECG_label_data['First_label']))
Y -= 1 # 从0开始
# 训练集测试集分割
train_x, test_x, train_y, test_y = train_test_split(X, Y,
                                                    test_size=1 - PARM.trainratio, 
                                                    random_state=0,
                                                    stratify=Y)
# 变成Torch.Tensor格式
train_x = t.FloatTensor(train_x) 
test_x = t.FloatTensor(test_x)
train_y = t.FloatTensor(train_y).type(t.LongTensor)
test_y = t.FloatTensor(test_y).type(t.LongTensor)
# 删除不用的数据(防止占内存)
del X, Y, ECG_feature_data, ECG_label_data, data, file
#%% Load Data 装载数据
# change to dataset
train_dataset = Data.TensorDataset(train_x, train_y)
# load data
loader = Data.DataLoader(
    dataset=train_dataset,      # torch TensorDataset format
    batch_size=PARM.batch_size, # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据
    )
#%%
cnn = ResNet34() # 创建神经网络    
# 放入到GPU
if PARM.ifGPU:
    cnn = cnn.cuda()
# print(cnn)
#%% Training Network 训练网络
## 测试集放入到GPU
if PARM.ifGPU:
    test_x = test_x.cuda()
    test_y = test_y.cuda()
## 优化算法
optimizer = t.optim.Adam(cnn.parameters(), lr=PARM.lr) # 使用adam算法来进行优化，lr是学习率，学习率越高，学习速度越快，越低，精度可能会越高
## 损失函数
loss_func = nn.CrossEntropyLoss() # 损失函数（对于多分类问题使用交叉熵）
## 迭代过程
print('Start Training')
for epoch in range(PARM.epoch): # 总体迭代次数
    for step, (train_x, train_y) in enumerate(loader): # 分配 batch data, normalize x when iterate train_loader
        if PARM.ifGPU:
            train_x = train_x.cuda()
            train_y = train_y.cuda()
        output = cnn(train_x)               # 输入x得到输出out
        loss = loss_func(output, train_y)   # 计算损失函数
        
        optimizer.zero_grad()           # 将梯度清成0        clear gradients for next train
        loss.backward()                 # 将损失误差反向传播，计算出梯度   backpropagation, compute gradients
        optimizer.step()                # 使用梯度，利用优化算法计算出新的网络    apply gradients
        # 计算准确率
        if step % 1 == 0:
            test_output = cnn(test_x) # 输入测试集得到输出test_out           
            if PARM.ifGPU:
                pred_train = t.max(output, 1)[1].cuda().data.squeeze()
                pred_test = t.max(test_output, 1)[1].cuda().data.squeeze()
            else:
                pred_train = t.max(output, 1)[1].data.squeeze()     # 训练集输出的分类结果
                pred_test = t.max(test_output, 1)[1].data.squeeze() # 测试集输出的分类结果
            # 计算当前训练和测试的准确率
            accuracy_train = float((pred_train == train_y.data).sum()) / float(train_y.size(0))
            accuracy_test = float((pred_test == test_y.data).sum()) / float(test_y.size(0))
            #输出
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data, '| train accuray: %.2f' % accuracy_train, '| test accuracy: %.2f' % accuracy_test)
print('End Training')
# %%
pre=pred_test.data.cpu().numpy()+1
# pre = np.array(pred_test) + 1
test =test_y.data.cpu().numpy()+1
# test = np.array(test_y.data) + 1

pre_dict = {'Recording': [], 'Result': []}
test_dict = {'Recording': [], 'First_label': []}

count = 0
for i in range(len(pre)):
    pre_dict['Recording'].append(count)
    pre_dict['Result'].append(pre[i])

    test_dict['Recording'].append(count)
    test_dict['First_label'].append(test[i])
    count += 1

pre = pd.DataFrame(pre_dict)
test = pd.DataFrame(test_dict)
# %%
test['Second_label'] = ''
test['Third_label'] = ''
# %%
pre.to_csv('1.csv', index=False)
test.to_csv('2.csv', index=False)
score('1.csv', '2.csv')
