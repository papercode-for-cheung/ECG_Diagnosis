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
#%% Input Arguments
parser = argparse.ArgumentParser(description='Experiment8(CNN_Transformer_LC): Train the model for diagnosing the heart disease by the ECG.')
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
    weight_list = [0.8,0.5,0.25,0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0]
    counter = 0
    for weight in weight_list:
        counter +=1
        print('w=', weight)
        cnn_transformer = models.CNN_Transformer().cuda() if Args.cuda else models.CNN_Transformer()
        # optimizer
        optimizer = torch.optim.Adam(cnn_transformer.parameters(), lr=Args.learn_rate)
        # loss function
        loss_func = nn.CrossEntropyLoss()
        # evaluate
        Accuracy = []
        F1 = []
        # %% ########## Training ##########
        if Args.show_plot:
            fig = plt.figure(counter)
            plt.ion()
        print('>>>>> Start Training')
        for epoch in range(Args.epoch):
            # load data
            loader_train = data_process.dataloader(train_x, train_y, Args.batch_size, shuffle=True)
            ##### Train #####
            for step, (x, y) in enumerate(loader_train):  # input batch data from train loader
                ##### learning #####
                output = []
                embedding = []
                cnn_transformer.train()
                for i in x:
                    input = i.cuda() if Args.cuda else i
                    output0, embedding0 = cnn_transformer(input)
                    output.append(output0)
                    embedding.append(embedding0)
                output = torch.cat(output)
                embedding = torch.cat(embedding)
                loss_lc = models.LinkConstraints(embedding, y, weight_decay=weight)
                y = torch.FloatTensor(y).type(torch.LongTensor).cuda() if Args.cuda else torch.FloatTensor(y).type(torch.LongTensor)
                loss = loss_lc + loss_func(output, y)  # get loss
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
                output = []
                cnn_transformer.eval()
                for i in x:
                    input = i.cuda() if Args.cuda else i
                    output0, _ = cnn_transformer(input)
                    output.append(output0)
                output = torch.cat(output)
                y = torch.FloatTensor(y).type(torch.LongTensor).cuda() if Args.cuda else torch.FloatTensor(y).type(torch.LongTensor)
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
                drawing.draw_result([Accuracy, F1], fig, ['Accuracy', 'F1'], True)
            del x, y, input, pred, output
            if Args.cuda: torch.cuda.empty_cache()  # empty GPU memory
        print('>>>>> End Training')
        # %% ########## Output and Save ##########
        print('>>>>> Save Result')
        pre = all_pred[0].data.cpu().numpy() + 1
        test = all_y[0].data.cpu().numpy() + 1
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
            plt.savefig("./Result/result.jpg")
    plt.ioff()
    plt.show()