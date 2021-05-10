# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/1 19:10
@Author  : QuYue
@File    : data_process.py
@Software: PyCharm
Introduction: Make data processing.
"""
#%% Import Packages
import torch
import torch.utils.data as Data
import numpy as np
from sklearn.model_selection import train_test_split
#%% Functions
def split(X, Y, Args, seed=1):
    # split the data to train and test
    train_x, test_x, train_y, test_y = train_test_split(X, Y,
                                                        test_size=1 - Args.train_ratio,
                                                        random_state=seed,
                                                        stratify=Y)
    return train_x, test_x, train_y, test_y

def data_loader(X, Y, batch_size):
    # load the data
    dataset = Data.TensorDataset(X, Y)
    loader = Data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    return loader
#%% Main Function
if __name__ == '__main__':
    class ARGS():
        def __init__(self):
            self.trainratio = 0.8
    Args = ARGS()
    # create data
    X = np.random.randn(20, 10)
    Y = np.random.randint(2, size=(20, 1))
    # data split
    train_x, test_x, train_y, test_y = split(X, Y, Args, seed=1)
    # change to tensor
    train_x = torch.FloatTensor(train_x)
    test_x = torch.FloatTensor(test_x)
    train_y = torch.FloatTensor(train_y).type(torch.LongTensor)
    test_y = torch.FloatTensor(test_y).type(torch.LongTensor)
    # load data
    train_loader = data_loader(train_x, train_y, 10)
    test_loader = data_loader(test_x, test_y, 2)
    #　import data from loader
    print('train')
    for i, j in train_loader:
        print(i.shape, j.shape)
    print('test')
    for i, j in test_loader:
        print(i.shape, j.shape)



