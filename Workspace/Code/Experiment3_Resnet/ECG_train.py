# -*- coding: utf-8 -*-
"""
@Time    : 2019/5/27 20:11
@File    : ECG_train.py
@Software: PyCharm
Introduction: Train the model for diagnosing the heart disease by the ECG.
"""
#%% Import Packages
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import read_config
import read_data
import data_process
import models
import score_py3
import drawing
#%% Input Arguments
parser = argparse.ArgumentParser(description='Experiment3(Resnet): Train the model for diagnosing the heart disease by the ECG.')
parser.add_argument('-c', '--config', type=str, default='./Config/config.ini', metavar='str',
                    help="the path of configure file (default: './Config/config.ini')")
Args = parser.parse_args() # the Arguments
Args = read_config.read(Args) # read configure file
if Args.cuda:
    print('Using GPU.')
else:
    print('Using CPU.')
#%% Main Function
if __name__ == '__main__':
    # %% ########## Read Data ##########
    print('>>>>> Read Data')
    ECG_data, ECG_label = read_data.extract_data(read_data.read_data(Args))  # read data
    # %% ########## Data Processing ##########
    print('>>>>> Data Processing')
    ECG_data = data_process.cut_out(ECG_data, Args.len)  # cut out the ECG signals
    ECG_data = data_process.axis_change(ECG_data)  # change the axis
    ECG_label = data_process.label_from_0(ECG_label)  # label from 0
    # split data
    train_x, test_x, train_y, test_y = data_process.train_test_split(ECG_data, ECG_label,
                                                                     trainratio=Args.trainratio,
                                                                     random_state=0)
    # change to Tensor
    train_x, train_y = data_process.to_tensor(train_x, train_y)
    test_x, test_y = data_process.to_tensor(test_x, test_y)
    # %% ########## Load Data ##########
    print('>>>>> Load Data')
    # change to dataset
    train_dataset = Data.TensorDataset(train_x, train_y)
    test_dataset = Data.TensorDataset(test_x, test_y)
    # load data
    loader_train = Data.DataLoader(
        dataset=train_dataset,  # torch TensorDataset format
        batch_size=Args.batch_size,  # mini batch size
        shuffle=True,  # shuffle or not
        num_workers=0,  # multi-worker for load data
    )
    loader_test = Data.DataLoader(test_dataset, Args.batch_size)
    del ECG_data, ECG_label, train_x, test_x, train_y, test_y
    # %% ########## Create model ##########
    print('>>>>> Create model')
    resnet = models.ResNet34().cuda() if Args.cuda else models.ResNet34()
    # optimizer
    optimizer = torch.optim.Adam(resnet.parameters(), lr=Args.learn_rate)
    # loss function
    loss_func = nn.CrossEntropyLoss()
    # evaluate
    Accuracy = []
    F1 = []
    # save
    Best_result = [0, [], []]
    # %% ########## Training ##########
    if Args.show_plot:
        fig = plt.figure(1)
        plt.ion()
    print('>>>>> Start Training')
    for epoch in range(Args.epoch):
        ##### Train #####
        for step, (x, y) in enumerate(loader_train):  # input batch data from train loader
            ##### learning #####
            if Args.cuda:
                x = x.cuda()
                y = y.cuda()
            resnet.train()
            output = resnet(x)
            loss = loss_func(output, y)  # get loss
            optimizer.zero_grad()  # clear gradients for backward
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            ##### train evaluate #####
            if Args.show_log:
                if step % 1 == 0:
                    if Args.cuda:
                        pred = torch.max(output, 1)[1].cuda().data.squeeze()
                    else:
                        pred = torch.max(output, 1)[1].data.squeeze()
                    # accuracy
                    accuracy_train = score_py3.accuracy(pred, y.data)
                    # F1
                    f1_train = score_py3.score_f1(pred, y.data)
                    # print
                    print('Epoch: %s | Train Accuracy: %.5f | Train F1: %.5f | Loss: %.2f' % (
                        epoch, accuracy_train, f1_train, loss.data))
        ##### Test #####
        all_y = []
        all_pred = []
        for step, (x, y) in enumerate(loader_test):
            if Args.cuda:
                x = x.cuda()
                y = y.cuda()
            resnet.eval()
            output = resnet(x)
            if Args.cuda:
                pred = torch.max(output, 1)[1].cuda().data.squeeze()
            else:
                pred = torch.max(output, 1)[1].data.squeeze()
            all_pred.append(pred)
            all_y.append(y.data)
        pred = torch.cat(all_pred)
        y = torch.cat(all_y)
        accuracy_test = score_py3.accuracy(pred, y.data)
        f1_test = score_py3.score_f1(pred, y.data)
        if Args.show_log:
            print('Epoch: %s | Train Accuracy: %.5f | Train F1: %.5f' % (epoch, accuracy_test, f1_test))
        Accuracy.append(accuracy_test)
        F1.append(f1_test)
        if Args.show_plot:
            drawing.draw_result([Accuracy, F1], fig, ['Resnet Accuracy', 'Resnet F1'], True)
        # save
        if f1_test >= Best_result[0]:
            Best_result[0] = f1_test
            Best_result[1] = pred.data.cpu().numpy() + 1
            Best_result[2] = y.data.cpu().numpy() + 1
        del x, y, pred, output
        if Args.cuda: torch.cuda.empty_cache()  # empty GPU memory
    print('>>>>> End Training')
    # %% ########## Output and Save ##########
    print('>>>>> Save Result')
    pre = Best_result[1]
    test = Best_result[2]
    ##### confusion matrix #####
    confmat = confusion_matrix(y_true=test, y_pred=pre)
    print(confmat)
    ##### save csv #####
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
    test['Second_label'] = ''
    test['Third_label'] = ''
    pre.to_csv('./Result/1.csv', index=False)
    test.to_csv('./Result/2.csv', index=False)
    score_py3.score('./Result/1.csv', './Result/2.csv')
    ##### save process #####
    log = pd.DataFrame([Accuracy, F1], index=['Accuracy', 'F1'])
    log.to_csv('./Result/log.csv')
    ##### save figure #####
    if Args.show_plot:
        plt.ioff()
        plt.savefig("./Result/result.jpg")
        plt.show()
