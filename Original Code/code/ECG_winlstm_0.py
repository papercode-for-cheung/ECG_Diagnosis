# -*- coding: utf-8 -*-
# %% Import Packages
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import scipy.io as scio
from sklearn.model_selection import train_test_split
from score_py3 import score_s,score
import utils1
import matplotlib.pyplot as plt
from sklearn .metrics import confusion_matrix
import torch.nn.functional as F
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
        # 数据参数
        self.datanum = 6877 # 提取数据量
        self.trainratio = 0.9  # 训练集的比例
        # 窗口参数
        self.winwidth = 3000  # 窗口宽度
        self.winstep = 1500  # 窗口移动步长
        # 训练参数
        self.batch_size = 100 # Batch的大小
        self.epoch =  150#迭代次数
        self.lr = 0.0008  # Learning rate 学习率，越大学习速度越快，越小学习速度越慢，但是精度可能会提高
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
        sample['data'] = load_data['ECG']['data'][0][0]  # ECG数据
        sample['data']
#        sample['data'] = sample['data'][:, :int(sample['data'].shape[1] // 2 * 2)]
#        sample['data'] = sample['data'].reshape(sample['data'].shape[0], -1, 2)
#        sample['data'] = np.average(sample['data'], axis=2)
        sample['data'] = utils1.remove_outlier(sample['data'])
        sample['data'] = utils1.remove_noise(sample['data'])
        ECG_feature_data.append(sample)  # 存入特征数据中
# 标签数据
ECG_label_data = pd.read_csv(PATH + LABEL)
ECG_label_data = ECG_label_data.iloc[0: PARM.datanum]  # 存入标签数据


# %% Reprocess Data 数据预处理
# 窗口切割
def get_windows(data, winwidth, winstep):
    # 窗口切割函数
    L = data.shape[1]  # 数据的长度
    windows = []
    for i in range(0, L - winwidth + 1, winstep):
        w = data[:, i: i + winwidth]
        windows.append(w)
    windows = np.array(windows)[..., np.newaxis]
    return windows

# 性别转换成特征
sex_change = {'Male': [1, 0], 'Female': [0, 1]}
# 年龄标准化
ages = np.array(ages)

def z_score(age, mean=ages.mean(), std=ages.std()):
    return (age - mean) / std

# 训练数据
X = []
for i in ECG_feature_data:
    sample = dict()
    sample['data'] = torch.FloatTensor(get_windows(i['data'], PARM.winwidth, PARM.winstep))
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

class CNN_Linear(nn.Module):
    def __init__(self):
        super(CNN_Linear, self).__init__()
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
        self.network1 = nn.Sequential(
                nn.Linear(256, 200),  # 卷积层后加一个普通的神经网络
                #nn.BatchNorm1d(256),
                nn.Dropout(0.7),
                nn.ReLU(),
                nn.Linear(200, 128),# 卷积层后加一个普通的神经网络
                #nn.BatchNorm1d(100),
                nn.Dropout(0.5),
                nn.ReLU(),
                )
        self.bilstm = nn.LSTM(128, 256, num_layers=1,batch_first=True, bidirectional=True)
        self.networks2 = nn.Sequential(
                nn.Linear(256*2,100),
                nn.Dropout(0.7),
                nn.Linear(100,9),
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
        x = x.reshape(x.shape[0], -1) # win_num * feature
        x = self.network1(x)
        x = x.unsqueeze(0)
        x = self.dropout(x)
        x, (h_n, c_n) = self.bilstm(self.relu(x))
        x = h_n[:2, :, :]
        x = h_n.reshape(1, -1)
       #  x = h_n[:-4,:,:].reshape(1,-1)# 取最后的一个双向层的隐含层的输出
       #  x =self.dropout(x)
        x = self.networks2(x)
        return x


#%%
acc = []
f1 = []
cnn_rnn = CNN_Linear()
plt.figure(1)
plt.ion()
if PARM.ifGPU:
    cnn_rnn = cnn_rnn.cuda()
optimizer = torch.optim.Adam(cnn_rnn.parameters(), lr=PARM.lr)
loss_func = nn.CrossEntropyLoss()
for epoch in range(PARM.epoch):
    for step, (x, y) in enumerate(loader):
        output = []
        for i in x:
            input = i['data'].cuda() if PARM.ifGPU else i['data']
            output.append(cnn_rnn(input))
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
#            F1 = score_s(pred, y)
            print('Epoch: %s |step: %s | train loss: %.2f | accuracy: %.2f '
                  % (epoch, step, loss.data, accuracy))
            del x, y, input, output
            if PARM.ifGPU: torch.cuda.empty_cache()
    #%%
    all_y = []
    all_pred = []
    for step, (x, y) in enumerate(test_loader):
        y = torch.FloatTensor(y).type(torch.LongTensor)
        if PARM.ifGPU:
            y = y.cuda()
        cnn_rnn.eval()  # test model
        output = []
        for i in x:
            input = i['data'].cuda() if PARM.ifGPU else i['data']
            output.append(cnn_rnn(input))
        output = torch.cat(output)
        cnn_rnn.train() # train model
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
    print('Epoch: %s | test accuracy: %.2f | f1: %.2f  ' % (epoch, accuracy,F1))
    acc.append(accuracy)
    f1.append(F1)
    drawing(acc, f1)
    del x, all_pred, all_y, input, output
    if PARM.ifGPU: torch.cuda.empty_cache() # empty GPU memory

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



