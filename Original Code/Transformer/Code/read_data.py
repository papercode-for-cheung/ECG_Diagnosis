# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/1 15:42
@Author  : QuYue
@File    : read_data.py
@Software: PyCharm
Introduction: Read the ECG data.
"""
#%% Import Packages
import numpy as np
import pandas as pd
import os
import scipy.io as scio
import matplotlib.pyplot as plt
#%% Functions
def read_train(Args):
    # Read the ECG train data
    ##### feature #####
    path = Args.train_data_path
    count = 0
    train_data = dict()
    for file in os.listdir(path):
        count +=1
        if count > Args.datanum: # control the number of read data
            break
        if file.endswith('.mat'):
            id = file.split('.')[0]
            load_data = scio.loadmat(path+'\\'+file) # Read.mat
            sample = load_data['data']
            train_data[id] = sample # Save to train_data
    ##### label #####
    path = Args.train_label_path
    train_label = pd.read_csv(path, sep='\t', names=['ID', 'label'])
    train_label = train_label.set_index('ID')
    ##### change to array #####
    ECG_train_data = []
    ECG_train_label = []
    for i in train_data.keys():
        ECG_train_data.append(train_data[i])
        ECG_train_label.append(train_label.loc[i]['label'])
    ECG_train_data = np.array(ECG_train_data) # change to array
    ECG_train_label = np.array(ECG_train_label) # change to array
    ##### add new axis
    ECG_train_data = ECG_train_data[..., np.newaxis]
    ECG_train_label = ECG_train_label[:, np.newaxis]
    return ECG_train_data, ECG_train_label

def read_test(Args):
    # Read the ECG test data
    path = Args.test_data_path
    ECG_test_data = []
    ECG_test_name = []
    for file in os.listdir(path):
        if file.endswith('.mat'):
            id = file.split('.')[0]
            load_data = scio.loadmat(path+'\\'+file) # Read .mat
            sample = load_data['data']
            ECG_test_data.append(sample) # Save to test_data
            ECG_test_name.append(id)
    # change to array
    ECG_test_data = np.array(ECG_test_data)
    ##### add new axis
    ECG_test_data = ECG_test_data[:, np.newaxis]
    return ECG_test_data, ECG_test_name

def show(data):
    # show one of the ECG data
    channel, length = data.shape
    plt.figure()
    for i in range(channel):
        signal = data[i]
        plt.subplot(3, 4, i+1)
        plt.plot(signal)
        plt.title('channel %s' %i)
    plt.show()
#%% Main Functions
if __name__ == '__main__':
    class ARGS():
        def __init__(self):
            self.train_data_path = r'../Data/preliminary/TRAIN'
            self.train_label_path = r'../Data/preliminary/reference.txt'
            self.test_data_path = r'../Data/preliminary/TEST'
            self.datanum = 10
    Args = ARGS()
    ECG_train_data, ECG_train_label = read_train(Args)
    ECG_test_data, ECG_test_name = read_test(Args)
    show(ECG_test_data[0])