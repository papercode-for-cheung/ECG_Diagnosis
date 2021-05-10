# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:45:08 2019

@author: my_lab
"""
#%%
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import scipy.io as scio
from sklearn.model_selection import train_test_split
from utils1 import *
import matplotlib.pyplot as plt
from score_py3 import score_s
#%% Set Hyper Parameter 超参数设置
class Parameter:
    # 超参数类
    def __init__(self):
        # 数据参数
        self.datanum = 6877# 提取数据量
        self.trainratio = 0.9  # 训练集的比例
        # 训练参数
        self.batch_size = 100 # Batch的大小
        self.epoch =  200 # 迭代次数
        self.lr = 0.00008 # Learning rate 学习率，越大学习速度越快，越小学习速度越慢，但是精度可能会提高
        self.ifGPU = False  # 是否需要使用GPU（需要电脑里有NVIDIA的独立显卡并安装CUDA驱动才可以)


PARM = Parameter()
PARM.ifGPU = True  # 开启GPU
PATH = 'E:\\zhumin\\TrainingSet1\\TrainingSet1\\'
LABEL = 'REFERENCE.csv'
# %% Input Data 数据输入
 # 特征数据
count = 0
ECG_feature_data = []
ages = []
for file in os.listdir(PATH):
    count += 1
    if count > PARM.datanum:
        break

    if file.endswith('.mat'):
        load_data = scio.loadmat(PATH + file)  # 读取mat格式数据
        sample = dict()  # 数据以字典形式存储
        sample['sex'] = str(load_data['ECG']['sex'][0][0][0])  # 性别
        try:
            sample['age'] = int(load_data['ECG']['age'][0][0][0][0])  # 年龄
        except:
            sample['age'] = 60
        ages.append(sample['age'])
        sample['data'] = load_data['ECG']['data'][0][0] 
        
        # ECG数据
#        sample['data'] = sample['data'][:,:int(sample['data'].shape[1]//4*4)]
#        sample['data'] = sample['data'].reshape(sample['data'].shape[0],-1,4)
#        sample['data'] = np.average(sample['data'],axis=2)
        sample['data'] = remove_outlier(sample['data'])
        sample['data'] = remove_noise(sample['data'])

        ECG_feature_data.append(sample)  # 存入特征数据中
# 标签数据
ECG_label_data = pd.read_csv(PATH + LABEL)
ECG_label_data = ECG_label_data.iloc[0: PARM.datanum]  # 存入标签数据
# %% Reprocess Data 数据预处理
# 性别转换成特征
sex_change = {'Male': [1, 0], 'Female': [0, 1]}
# 年龄标准化
ages = np.array(ages)

def z_score(age, mean=ages.mean(), std=ages.std()):
    return (age - mean) / std
def get_windows(data, winwidth, winstep):
    # 窗口切割函数
    L = data.shape[1]  # 数据的长度
    windows = []
    for i in range(0, L - winwidth + 1, winstep):
        w = data[:, i: i + winwidth]
        windows.append(w)
    windows = np.array(windows)[..., np.newaxis]
    return windows
# 训练数据
X = []
for i in ECG_feature_data:
    sample = dict()
    sample['data'] = torch.FloatTensor(get_windows(i['data'],3000,1500))
    sample['sex'] = sex_change[i['sex']]
    sample['age'] = z_score(i['age'])
    X.append(sample)
# 测试数据
Y = np.array(list(ECG_label_data['First_label']))
Y -= 1  # 从0开始
# 训练集测试集分割
train_x, test_x, train_y, test_y = train_test_split(X, Y,
                                                    test_size=1 - PARM.trainratio,
                                                    random_state=0,
                                                    stratify=Y)
# 标签变成LongTensor格式
train_y = torch.FloatTensor(train_y).type(torch.LongTensor)
test_y = torch.FloatTensor(test_y).type(torch.LongTensor)
# 删除不用的数据(防止占内存)
del X, Y, ECG_feature_data, ECG_label_data, ages, file, load_data, sample, sex_change
# %% Load Data 装载数据
def dataloader(feature, label, batch_size=1, shuffle=True):
    L = len(feature)
    dataset = []
    for i in range(L):
        dataset.append([feature[i], label[i]])
    if shuffle:
        np.random.shuffle(dataset)
    batch = []
    for i in range(0, L, batch_size):
        f = [j[0] for j in dataset[i: i + batch_size]]
        l = [j[1] for j in dataset[i: i + batch_size]]
        batch.append([f, l])
    return batch


# load data
loader = dataloader(train_x, train_y, PARM.batch_size, True)
test_loader = dataloader(test_x, test_y, 100, True)
#%% Create Network 网络构建
class CNN_Linear_RNN4(nn.Module):
    def __init__(self):
        super(CNN_Linear_RNN4, self).__init__()
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
        # 第二个卷积层（包括了卷积、整流、池化）
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (10, 1), 1, 0),  # 直接简写了，参数的顺序同上
            nn.ReLU(),
            nn.MaxPool2d((3, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (10, 1), 1, 0),  # 直接简写了，参数的顺序同上
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
        self.bilstm =nn.LSTM(128,256,batch_first=True,bidirectional=True)
        self.network1 = nn.Sequential(
                nn.Linear(256, 200),  # 卷积层后加一个普通的神经网络
                #nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(200, 128),# 卷积层后加一个普通的神经网络
                #nn.BatchNorm1d(100),
                nn.Dropout(0.5),
                nn.ReLU(),
                )
        self.bilstm2 =nn.LSTM(512,512,bidirectional=True)

        self.network3 = nn.Sequential(
                nn.Linear(512, 100),
                # nn.BatchNorm1d(64),
                nn.Dropout(0.7),
                nn.Linear(100, 4),
                )
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x =self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.reshape(x.shape[0], -1) # win_num * feature
        x = self.network1(x)
        x = x.unsqueeze(0)
        x = self.dropout(x)
        x, (h_n, c_n) = self.bilstm(self.relu(x))
        x = h_n[0, :, :]
        x = h_n.reshape(1, -1)
       #  x = h_n[:-4,:,:].reshape(1,-1)# 取最后的一个双向层的隐含层的输出
        x = self.network3(x)
        return x
                 # 31 * win_num * 256
                 # 2 * win_num * 512
#        print(h_n.shape)
#        x = h_n.permute(1, 0, 2)           # win_num * 2 * 512
#        x = x.reshape(x.shape[0], -1)      # win_num * 1024
#        x = self.network3(x)
#        return x
#%%
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(16,64)
        self.networks = nn.Sequential(
                nn.Linear(64,64),
                nn.Dropout(0.5),
                nn.Linear(64,9)
                )
    def forward(self,x):
        x = x.permute(1,0,2) 
        x,(h_n,c_n)= self.lstm(x)
        x = h_n[0, :, :]
#        print(h_n.shape)
        x = self.networks(x)
        return x
#%%
cnn_rnn1 = CNN_Linear_RNN4()
cnn_rnn2 = CNN_Linear_RNN4()
cnn_rnn3 = CNN_Linear_RNN4()
cnn_rnn4 = CNN_Linear_RNN4()
lstm =LSTM()
if PARM.ifGPU:
    cnn_rnn1 = cnn_rnn1.cuda()
    cnn_rnn2 = cnn_rnn2.cuda()
    cnn_rnn3 = cnn_rnn3.cuda()
    cnn_rnn4 = cnn_rnn4.cuda()
    lstm =LSTM().cuda()
#%%
optimizer = torch.optim.Adam([{'params': cnn_rnn1.parameters()},
                              {'params': cnn_rnn2.parameters()},
                              {'params': cnn_rnn3.parameters()},
                              {'params': cnn_rnn4.parameters()},
                              {'params': lstm.parameters()}], lr=PARM.lr)
loss_func = nn.CrossEntropyLoss()
    
#%%
acc = []
f1 = []
plt.figure(1)
plt.ion()
for epoch in range(PARM.epoch):
    for step, (x, y) in enumerate(loader):
        output = []
        for i in x: 
            input = i['data'].cuda() if PARM.ifGPU else i['data']
            result1 = cnn_rnn1(input)
            result2 = cnn_rnn2(input)
            result3 = cnn_rnn3(input)
            result4 = cnn_rnn4(input)
            results = torch.cat((result1,result2,result3,result4),dim=1)
            l = len(results)
            if l<45:
                    z = torch.zeros((45-l,16)).cuda()
                    results = torch.cat((z,results),dim=0)
            else:
                results = results[:45,:]
            result = lstm(results[np.newaxis,:,:])
            output.append(result)
        output = torch.cat(output)
        y = torch.FloatTensor(y).type(torch.LongTensor)
        if PARM.ifGPU:
            y = y.cuda()
        loss = loss_func(output, y)  # loss
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        if step % 1 == 0:
            if PARM.ifGPU:
                pred = torch.max(output, 1)[1].cuda().data.squeeze()
            else:
                pred = torch.max(output, 1)[1].data.squeeze()
            accuracy = float((pred == y).sum()) / float(y.size(0))
            # F1 = score_s(pred, y)
            print('Epoch: %s |step: %s | train loss: %.2f | accuracy: %.2f '
                  % (epoch, step, loss.data, accuracy))
            del x, y, input, output
            if PARM.ifGPU: torch.cuda.empty_cache()
    num = 0
    correct_num = 0   
    all_y = []
    all_pred = []
    for step, (x, y) in enumerate(test_loader): 
        # cnn_rnn1.eval()
        # cnn_rnn2.eval()
        # cnn_rnn3.eval()
        # cnn_rnn4.eval()# test model
        output = []
        for i in x:
            input = i['data'].cuda() if PARM.ifGPU else i['data']
            result1 = cnn_rnn1(input)
            result2 = cnn_rnn2(input)
            result3 = cnn_rnn3(input)
            result4 = cnn_rnn4(input)
            results = torch.cat((result1,result2,result3,result4),dim=1)
            l = len(results)
            if l<45:
                    z = torch.zeros((45-l,16)).cuda()
                    results = torch.cat((z,results),dim=0)
            else:
                results = results[:45,:]
            result = lstm(results[np.newaxis,:,:])
            output.append(result)
        output = torch.cat(output)
        y = torch.FloatTensor(y).type(torch.LongTensor)
        if PARM.ifGPU:
            y = y.cuda()
        if PARM.ifGPU:
            pred = torch.max(output, 1)[1].cuda().data.squeeze()
        else:
            pred = torch.max(output, 1)[1].data.squeeze()
        all_y.append(y)
        all_pred.append(pred)
    # evaluate
    y = torch.cat(all_y)
    pred = torch.cat(all_pred)
    accuracy = float((pred== y).sum()) / float(y.size(0))
    F1 = score_s(pred, y)
    print('Epoch: %s | test accuracy: %.2f|f1: %.2f'% (epoch, accuracy,F1))
    del x, y, all_pred, all_y, input, output
    if PARM.ifGPU: torch.cuda.empty_cache() # empty GPU memory