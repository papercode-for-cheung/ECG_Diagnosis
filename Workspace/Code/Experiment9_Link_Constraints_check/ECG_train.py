# -*- coding: utf-8 -*-
"""
@Time    : 2019/5/27 21:56
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
#%% Function
def training(output, loss, optimizer, y, name, Args):
    optimizer.zero_grad()  # clear gradients for backward
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
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
            print('Epoch: %s | %s Train Accuracy: %.5f | Train F1: %.5f | Loss: %.2f' % (
                epoch, name, accuracy_train, f1_train, loss.data))

def Testing(pred, y, name):
    accuracy_test = score_py3.accuracy(pred, y.data)
    f1_test = score_py3.score_f1(pred, y.data)
    if Args.show_log:
        print('Epoch: %s | %s Train Accuracy: %.5f | Train F1: %.5f' % (epoch, name, accuracy_test, f1_test))
    return f1_test

#%% Input Arguments
parser = argparse.ArgumentParser(description='Experiment5(CNN_BiLSTM_LC): Train the model for diagnosing the heart disease by the ECG.')
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
    ECG_data = data_process.split_wins(ECG_data, Args.win_size, Args.win_step)  # split windows
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
    # loader_train = data_process.dataloader(train_x, train_y, Args.batch_size, shuffle=True)
    loader_test = data_process.dataloader(test_x, test_y, Args.batch_size, shuffle=True)
    del ECG_data, ECG_label
    # %% ########## Create model ##########
    print('>>>>> Create model')
    cnn_bilstm = models.CNN_BiLSTM().cuda() if Args.cuda else models.CNN_BiLSTM()
    cnn_bilstm0 = models.CNN_BiLSTM().cuda() if Args.cuda else models.CNN_BiLSTM()
    cnn_bilstm1 = models.CNN_BiLSTM().cuda() if Args.cuda else models.CNN_BiLSTM()
    cnn_bilstm2 = models.CNN_BiLSTM().cuda() if Args.cuda else models.CNN_BiLSTM()
    # optimizer
    optimizer = torch.optim.Adam(cnn_bilstm.parameters(), lr=Args.learn_rate)
    optimizer0 = torch.optim.Adam(cnn_bilstm0.parameters(), lr=Args.learn_rate)
    optimizer1 = torch.optim.Adam(cnn_bilstm1.parameters(), lr=Args.learn_rate)
    optimizer2 = torch.optim.Adam(cnn_bilstm2.parameters(), lr=Args.learn_rate)
    # loss function
    loss_func = nn.CrossEntropyLoss()
    # evaluate
    Accuracy = []
    F1 = []
    Accuracy0 = []
    F1_0 = []
    Accuracy1 = []
    F1_1 = []
    Accuracy2 = []
    F1_2 = []

    # %% ########## Training ##########
    if Args.show_plot:
        fig = plt.figure(1)
        plt.ion()
    print('>>>>> Start Training')
    for epoch in range(Args.epoch):
        # load data
        loader_train = data_process.dataloader(train_x, train_y, Args.batch_size, shuffle=True)
        ##### Train #####
        for step, (x, y) in enumerate(loader_train):  # input batch data from train loader
            ##### learning #####
            output = []
            output0 = []
            output1 = []
            output2 = []

            embedding0 = []
            embedding1 = []
            embedding2 = []
            cnn_bilstm.train()
            cnn_bilstm0.train()
            cnn_bilstm1.train()
            cnn_bilstm2.train()
            for i in x:
                input = i.cuda() if Args.cuda else i
                t_output, _ = cnn_bilstm(input)
                t_output0, t_embedding0 = cnn_bilstm0(input)
                t_output1, t_embedding1 = cnn_bilstm1(input)
                t_output2, t_embedding2 = cnn_bilstm2(input)
                # append
                output.append(t_output)

                output0.append(t_output0)
                embedding0.append(t_embedding0)
                output1.append(t_output1)
                embedding1.append(t_embedding1)
                output2.append(t_output2)
                embedding2.append(t_embedding2)

            output = torch.cat(output)
            output0 = torch.cat(output0)
            embedding0 = torch.cat(embedding0)
            output1 = torch.cat(output1)
            embedding1 = torch.cat(embedding1)
            output2 = torch.cat(output2)
            embedding2 = torch.cat(embedding2)


            loss_lc0 = models.LinkConstraints(embedding0, y, weight_decay=Args.link_weight)
            loss_lc1 = models.LinkConstraints1(embedding1, y, weight_decay=Args.link_weight)
            loss_lc2 = models.LinkConstraints2(embedding2, y, weight_decay=Args.link_weight)
            y = torch.FloatTensor(y).type(torch.LongTensor).cuda() if Args.cuda else torch.FloatTensor(y).type(torch.LongTensor)

            loss = loss_func(output, y)
            training(output, loss, optimizer, y,  'BiLSTM', Args)
            loss0 = loss_lc0 + loss_func(output0, y)
            training(output0, loss0, optimizer0, y,  'LC', Args)
            loss1 = loss_lc1 + loss_func(output1, y)
            training(output1, loss1, optimizer1, y, 'LC-1', Args)
            loss2 = loss_lc2 + loss_func(output2, y)
            training(output2, loss2, optimizer2, y, 'LC+1', Args)

        ##### Test #####
        all_y = []
        all_pred = []
        all_pred0 = []
        all_pred1 = []
        all_pred2 = []
        for step, (x, y) in enumerate(loader_test):
            output = []
            output0 = []
            output1 = []
            output2 = []
            cnn_bilstm.eval()
            cnn_bilstm0.eval()
            cnn_bilstm1.eval()
            cnn_bilstm2.eval()
            for i in x:
                input = i.cuda() if Args.cuda else i
                t_output, _ = cnn_bilstm(input)
                t_output0, _ = cnn_bilstm0(input)
                t_output1, _ = cnn_bilstm1(input)
                t_output2, _ = cnn_bilstm2(input)
                output.append(t_output)
                output0.append(t_output0)
                output1.append(t_output1)
                output2.append(t_output2)
            output = torch.cat(output)
            output0 = torch.cat(output0)
            output1 = torch.cat(output1)
            output2 = torch.cat(output2)
            y = torch.FloatTensor(y).type(torch.LongTensor).cuda() if Args.cuda else torch.FloatTensor(y).type(torch.LongTensor)
            if Args.cuda:
                pred = torch.max(output, 1)[1].cuda().data.squeeze()
                pred0 = torch.max(output0, 1)[1].cuda().data.squeeze()
                pred1 = torch.max(output1, 1)[1].cuda().data.squeeze()
                pred2 = torch.max(output2, 1)[1].cuda().data.squeeze()
            else:
                pred = torch.max(output, 1)[1].data.squeeze()
                pred0 = torch.max(output, 1)[1].data.squeeze()
                pred1 = torch.max(output, 1)[1].data.squeeze()
                pred2 = torch.max(output, 1)[1].data.squeeze()
            all_pred.append(pred)
            all_pred0.append(pred0)
            all_pred1.append(pred1)
            all_pred2.append(pred2)
            all_y.append(y.data)
        pred = torch.cat(all_pred)
        pred0 = torch.cat(all_pred0)
        pred1 = torch.cat(all_pred1)
        pred2 = torch.cat(all_pred2)
        y = torch.cat(all_y)
        F1.append(Testing(pred, y ,'BiLSTM'))
        F1_0.append(Testing(pred0, y, 'LC'))
        F1_1.append(Testing(pred1, y, 'LC-1'))
        F1_2.append(Testing(pred2, y, 'LC+1'))
        if Args.show_plot:
            drawing.draw_result([F1, F1_0, F1_1, F1_2], fig, ['BiLSTM', 'LC', 'LC-1', 'LC+1'], True)

        del x, y, pred, input, output
        if Args.cuda: torch.cuda.empty_cache()  # empty GPU memory
    print('>>>>> End Training')
    ##### save figure #####
    log = pd.DataFrame([F1, F1_0, F1_1, F1_2], index = ['F1', 'F1_0', 'F1_1', 'F1_2'])
    log.to_csv('./Result/log.csv')
    ##### save figure #####
    if Args.show_plot:
        plt.ioff()
        plt.savefig("./Result/result.jpg")
        plt.show()