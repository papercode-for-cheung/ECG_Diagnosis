# -*- coding: utf-8 -*-
"""
@Time    : 2019/5/25 16:37
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

def cut_out(X, len):
    # Cut out the ECG signal
    X_new = []
    for i in X:
        X_new.append(i[:, : len])
    output = np.array(X_new)
    return X_new

def axis_change(X):
    # Change the ECG data's axis to channel * length* width
    for i in range(len(X)):
        X[i] = X[i][..., np.newaxis]
    return X

def to_tensor(X, Y):
    # Change the data to torch.Tensor
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y).type(torch.LongTensor)
    return X, Y


def train_test_split(X, Y, trainratio=0.9, random_state=0):
    # Split the train data and test data
    train_x, test_x, train_y, test_y  = sklearn.model_selection.train_test_split(X, Y,
                                                                                 test_size=1 - trainratio,
                                                                                 random_state=random_state,
                                                                                 stratify=Y)
    return train_x, test_x, train_y, test_y

