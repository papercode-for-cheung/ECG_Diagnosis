# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/1 15:42
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
from math import isnan
#%% Functions
class ECG_DATA():
    # Class for ecg data
    def __init__(self, id):
        self.id = id
        self.sex = 'Male'
        self.age = 0
        self.ecg = np.zeros([12, 1])
        self.label = [0]

def multi_label_fuse(series):
    # Fuse the multi_labels(np.Series) to a list
    label = []
    for i in series:
        if not isnan(i):
            label.append(int(i))
    return label

def read_data(Args):
    # Read the ECG train data
    ##### feature #####
    path = Args.data_path
    count = 0
    data = dict()
    for file in os.listdir(path):
        count +=1
        if count > Args.datanum: # control the number of read data
            break
        if file.endswith('.mat'):
            id = file.split('.')[0]
            load_data = scio.loadmat(path+'/'+file) # Read.mat
            sample = ECG_DATA(id)
            sample.sex = load_data['ECG'][0][0][0][0]
            sample.age = int(load_data['ECG'][0][0][1][0][0]) if not isnan(load_data['ECG'][0][0][1][0][0]) else float('nan')
            sample.ecg = load_data['ECG'][0][0][2]
            data[id] = sample # Save to data
    ##### label #####
    path = Args.label_path
    label = pd.read_csv(path)
    label = label.set_index('Recording')
    ##### change to array #####
    ECG_data = []
    for i in data.keys():
        labels = multi_label_fuse(label.loc[i])
        data[i].label = labels
        ECG_data.append(data[i])
    return ECG_data

def extract_data(ECG_data):
    # Extract Data (feature: only ECG, label: only first label)
    feature = []
    label = []
    for i in ECG_data:
        feature.append(i.ecg)
        label.append(i.label[0])
    return feature, label

def show(data):
    # Show one of the ECG data (np.ndarray: channel * length)
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
            self.data_path = r"E:\zhumin\TrainingSet1\TrainingSet1"
            self.label_path = r'E:\zhumin\TrainingSet1\TrainingSet1\REFERENCE.csv'
            self.datanum = 2000
    Args = ARGS()
    ECG_data= read_data(Args)
    show(ECG_data[0].ecg)
    feature, label = extract_data(ECG_data)
