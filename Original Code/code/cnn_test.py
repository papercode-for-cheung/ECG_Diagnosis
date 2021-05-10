# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 00:11:00 2018

@author:zhumin
"""
# %% Import Package 加载模块
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import numpy as np
import pandas as pd
import os
import scipy.io as scio
from sklearn.model_selection import train_test_split
from score_py3 import score_s,score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#%%
def drawing(acc, f1):
    plt.cla()
    plt.plot(acc)
    plt.plot(f1)
    plt.legend(['accuracy', 'F1'], loc='lower right')
    plt.draw()
    plt.pause(0.01)
# %% Set Hyper Parameter 超参数设置
class Parameter:
    # 超参数类
    def __init__(self):
        self.batch_size =100# Batch的大小
        self.epoch = 40  # 迭代次数
        self.lr = 0.001  # Learning rate 学习率，越大学习速度越快，越小学习速度越慢，但是精度可能会提高
        self.ifGPU = False  # 是否需要使用GPU（需要电脑里有NVIDIA的独立显卡并安装CUDA驱动才可以)
        self.trainratio = 0.9  # 训练集的比例
PARM = Parameter()
PARM.ifGPU = True  # 开启GPU
PARM.datanum = 6877  # 提取数据量
PARM.len = 3000  # 数据截断长度
PATH =  'E:\\zhumin\\TrainingSet1\\TrainingSet1\\'
#PATH1 = '/home/yanghaoyu/文档/t/'
LABEL = 'REFERENCE.csv'
# %% Input Data
# 特征数据
count = 0
ECG_feature_data = []
Id = []
for file in os.listdir(PATH):
    count += 1
    if count > PARM.datanum:
        break
    Id.append(int(file.split('.')[0][1:]) - 1)
    data = scio.loadmat(PATH + file)
    data = data['ECG']['data'][0][0]
    ECG_feature_data.append(data)
# 标签数据
ECG_label_data = pd.read_csv(PATH+ LABEL)
ECG_label_data = ECG_label_data.iloc[Id]
# %% Reprocess Data 数据预处理
X = np.zeros([PARM.datanum, 12, PARM.len])
for i in range(len(ECG_feature_data)):
    X[i, :] = ECG_feature_data[i][:, :PARM.len]
X = X[:, np.newaxis,:, :]
Y = np.array(list(ECG_label_data['First_label']))
Y -= 1  # 从0开始
 #训练集测试集分割
train_x, test_x, train_y, test_y = train_test_split(X, Y,
                                                    test_size=1 - PARM.trainratio,
                                                    random_state=0,
                                                    stratify=Y)
# 变成Torch.Tensor格式
train_x = torch.FloatTensor(train_x)
test_x = torch.FloatTensor(test_x)
train_y = torch.FloatTensor(train_y).type(torch.LongTensor)
test_y = torch.FloatTensor(test_y).type(torch.LongTensor)

# 删除不用的数据(防止占内存)
del X, ECG_feature_data, ECG_label_data, data, file
# %% Load Data 装载数据


# change to dataset
train_dataset = Data.TensorDataset(train_x, train_y)
test_dataset = Data.TensorDataset(test_x, test_y)
# load data
loader = Data.DataLoader(
    dataset=train_dataset,  # torch TensorDataset format
    batch_size=PARM.batch_size,  # mini batch size
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
    num_workers=0,  # 多线程来读数据
)
loader1 = Data.DataLoader(test_dataset,100)
# %% Build CNN 搭建CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层（包括了卷积、整流、池化）
        self.conv1 = nn.Sequential(
            # 卷积
            nn.Conv2d(
                in_channels=1,  # 输入通道
                out_channels=32,  # 输出通道
                kernel_size=(12, 3),  # 卷积核的长度
                stride=1,  # 步长 （卷积核移动的步长）
                padding=0,  # 补0 zero_padding （对图片的边缘加上一圈0，防止越卷积图片越小）
            ),
            # 整流 使用ReLU函数进行整流
            nn.ReLU(),
            # 池化 使用最大池化
            nn.MaxPool2d(kernel_size=(1, 2))  # 最大池化 在3*1的范围内取最大
        )
        # 第二个卷积层（包括了卷积、整流、池化）
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(256, 256, (3, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )
        self.network = nn.Sequential(
            nn.Linear(768, 100),  # 卷积层后加一个普通的神经网络
            nn.Dropout(0.5),
            nn.Linear(100, 60),  # 卷积层后加一个普通的神经网络
            nn.Dropout(0.5),
            nn.Linear(60, 9),  # 卷积层后加一个普通的神经网络
        )
    def forward(self, x):
        # 输入（Batch_size * 12 * 4000 * 1）
        x = self.conv1(x)  # （Batch_size *  1330 *  * 1）
        x = x.permute(0,1,3,2)
        x = self.conv2(x)  # （Batch_size * 64 * 440 * 1）
        x = self.conv3(x)  # （Batch_size * 64 * 440 * 1）

        x = self.conv4(x)  # （Batch_size * 64 * 440 * 1）
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        # print(x.shape)
        x = self.conv8(x)
        x = self.conv9(x)
        x = x.view(x.size(0), -1)  # 将数据展平 (Batch_size * 28160)
        output = self.network(x)  # 输入到神经网络 (Batch_size * 1000)
        return output
# %% Create CNN
cnn = CNN()  # 创建神经网络
# 放入到GPU
if PARM.ifGPU:
    cnn = cnn.cuda()
# print(cnn)
    # %% Training Network 训练网络
## 测试集放入到GPU
if PARM.ifGPU:
    test_x = test_x.cuda()
    test_y = test_y.cuda()
## 优化算法
optimizer = torch.optim.Adam(cnn.parameters(), lr=PARM.lr)  # 使用adam算法来进行优化，lr是学习率，学习率越高，学习速度越快，越低，精度可能会越高
## 损失函数
loss_func = nn.CrossEntropyLoss()  # 损失函数（对于多分类问题使用交叉熵）
## 迭代过程
#%%
acc = []
f1 = []
plt.figure(1)
plt.ion()
print('Start Training')
for epoch in range(PARM.epoch):  # 总体迭代次数
    for step, (train_x, train_y) in enumerate(loader):  # 分配 batch data, normalize x when iterate train_loader
        if PARM.ifGPU:
            train_x = train_x.cuda()
            train_y = train_y.cuda()
        output = cnn(train_x)  # 输入x得到输出out
        loss = loss_func(output, train_y)  # 计算损失函数
        optimizer.zero_grad()  # 将梯度清成0        clear gradients for next train
        loss.backward()  # 将损失误差反向传播，计算出梯度   backpropagation, compute gradients
        optimizer.step()  # 使用梯度，利用优化算法计算出新的网络    apply gradients
        # 计算准确率
        if step % 1 == 0:
            if PARM.ifGPU:
                pred_train = torch.max(output, 1)[1].cuda().data.squeeze()

            else:
                pred_train = torch.max(output, 1)[1].data.squeeze()  # 训练集输出的分类结果
            # 计算当前训练和测试的准确率
            accuracy_train = float((pred_train == train_y.data).sum()) / float(train_y.size(0))

            # 输出
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data, '| train accuray: %.2f' % accuracy_train)
            #     '| test accuracy: %.2f' % accuracy_test)

    all_y = []
    all_pred = []
    for step, (x, y) in enumerate(loader1):
        if PARM.ifGPU:
            x = x.cuda()
            y = y.cuda()
        cnn.eval()
        test_out = cnn(x)
        pred_test = torch.max(test_out, 1)[1].data.squeeze()  # 训练集输出的分类结果
        if PARM.ifGPU:
            pred_test = torch.max(test_out, 1)[1].cuda().data.squeeze()
            # 计算当前测试的准确率
        all_pred.append(pred_test)
        all_y.append(y)

        # correct_num += int((pred_test == y.data).sum())
        # accuracy_test =float((pred_test == y.data).sum()) / float(y.size(0))
        # F1 = score_s(pred_test, y)
        # print('Epoch: %s | test accuracy: %.2f | f1: %.2f  ' % (epoch, accuracy_test, F1))
    # evaluate
    y = torch.cat(all_y)
    pred = torch.cat(all_pred)
    accuracy = float((pred == y).sum()) / float(y.size(0))
    F1 = score_s(pred, y)
    print('Epoch: %s | test accuracy: %.2f | f1: %.2f  ' % (epoch, accuracy, F1))
    acc.append(accuracy)
    f1.append(F1)
    drawing(acc, f1)
    del x,
    if PARM.ifGPU: torch.cuda.empty_cache()  # empty GPU memory
# %%
# %%
pre = pred.data.cpu().numpy() + 1
# pre = np.array(pred_test) + 1
test = y.data.cpu().numpy() + 1
# test = np.array(test_y.data) + 1
confmat = confusion_matrix(y_true = test, y_pred = pre)
print(confmat)
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
plt.ioff()
plt.show()












