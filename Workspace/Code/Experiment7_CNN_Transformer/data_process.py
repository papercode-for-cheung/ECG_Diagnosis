# -*- coding: utf-8 -*-
"""
@Time    : 2019/5/27 22:51
@File    : data_process.py
@Software: PyCharm
Introduction: Process the data.
"""
#%% Import Packages
import torch
import numpy as np
import torch
import sklearn.model_selection
#%% Function
def label_from_0(Y):
    # Change the label from 0
    Y = np.array(Y)
    Y -= Y.min()
    return Y

def split_wins(X, win_size, win_step):
    # Split the windows
    X_new = []
    for x in X:
        L = x.shape[1]  # length of the data
        windows = []
        for i in range(0, L - win_size + 1, win_step):
            w = x[:, i: i + win_size]
            windows.append(w)
        windows = np.array(windows)[..., np.newaxis]
        X_new.append(windows)
    return X_new

def train_test_split(X, Y, trainratio=0.9, random_state=0):
    # Split the train data and test data
    train_x, test_x, train_y, test_y  = sklearn.model_selection.train_test_split(X, Y,
                                                                                 test_size=1 - trainratio,
                                                                                 random_state=random_state,
                                                                                 stratify=Y)
    return train_x, test_x, train_y, test_y

def to_tensor(X, Y):
    # Change the data to torch.Tensor
    X_new = []
    for i in X:
        X_new.append(torch.FloatTensor(i))
    Y_new = torch.FloatTensor(Y).type(torch.LongTensor)
    return X_new, Y_new

def dataloader(feature, label, batch_size=1, shuffle=True):
    # Load Data
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


#%% Main Function
if __name__ == '__main__':
    a = np.random.randn(3, 14)
    b = np.random.randn(3, 17)
    c = [a, b]
    t = split_wins(c, 5, 3)