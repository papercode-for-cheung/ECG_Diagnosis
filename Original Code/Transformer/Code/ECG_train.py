# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/1 15:32
@Author  : QuYue
@File    : ECG_train.py
@Software: PyCharm
Introduction: Main Function for train the model for diagnosis by ECG
"""
#%% Import Packages
import torch
import arguments
import read_data
import data_process
import model
import score
import matplotlib.pyplot as plt
import drawing

#%% Get Arguments
train_args = {
    'cuda': True,
    'draw': True,
    'datanum': 600,
    'epoch': 200,
    'batch_size': 10,
    'train_ratio': 0.8,
    'learn_rate': 0.00008,
}
# 0.00008 batch_size=10 可以达到90.8 # 0.001 batch_size=10, model3可以达到91.8%
Args = arguments.get()
arguments.add(Args, train_args) # add the train_args
Args.cuda = Args.cuda and torch.cuda.is_available()
print('Using GPU.') if Args.cuda else print('Using CPU.')
Args.train_data_path = r'../Data/preliminary/TRAIN' # add paths
Args.train_label_path = r'../Data/preliminary/reference.txt'
Args.test_data_path = r'../Data/preliminary/TEST'
torch.manual_seed(Args.seed)
if Args.cuda:
    torch.cuda.manual_seed(Args.seed)
#%% Main Function
if __name__ == '__main__':
    #%%　Read Data
    print('==>Read Data')
    ECG_train_data, ECG_train_label = read_data.read_train(Args)
    # read_data.show(ECG_train_data[0])
    #%% Data Processing
    print('==>Data Processing')
    # data split
    train_x, valid_x, train_y, valid_y = data_process.split(ECG_train_data, ECG_train_label,
                                                            Args, seed=1)
    # change to tensor
    train_x = torch.FloatTensor(train_x)
    valid_x = torch.FloatTensor(valid_x)
    train_y = torch.FloatTensor(train_y).type(torch.LongTensor)
    valid_y = torch.FloatTensor(valid_y).type(torch.LongTensor)
    # load data
    train_loader = data_process.data_loader(train_x, train_y, Args.batch_size)
    valid_loader = data_process.data_loader(valid_x, valid_y, 30)
    # empty cache
    del ECG_train_data, ECG_train_label,  train_x, valid_x, train_y, valid_y
    #%%
    print('==>Training Model')
    diagnosis = model.Diagnosis2()
    optimizer = torch.optim.Adam(diagnosis.parameters(), lr=Args.learn_rate) # optimizer
    loss_func = torch.nn.CrossEntropyLoss()  # 损失函数（交叉熵）
    criterion = torch.nn.L1Loss()
    if Args.draw:
        plt.ion()
        figure = plt.figure(1, figsize=(10, 6))
        F1_list = []
        acc_list = []
    if Args.cuda:
        diagnosis = diagnosis.cuda()
    for epoch in range(Args.epoch):
        # Training
        for step, (x, y) in enumerate(train_loader):
            y = torch.squeeze(y) # delete a axis
            if Args.cuda:
                x, y = x.cuda(), y.cuda()
            diagnosis.train()  # train model
            output = diagnosis(x)
            loss = loss_func(output, y)  # loss
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  #  backpropagation, compute gradients
            optimizer.step()  # apply gradients
            if step % 1 == 0:
                if Args.cuda:
                    pred = torch.max(output, 1)[1].cuda().data.squeeze()
                else:
                    pred = torch.max(output, 1)[1].data.squeeze()
                # evaluate
                accuracy = score.accuracy(pred, y)
                F1 = score.F1(pred, y)
                print('Epoch: %s |step: %s | train loss: %.2f | accuracy: %.2f | F1: %.4f' %(epoch, step, loss.data, accuracy, F1))
        #%% Testing
        all_y = []
        all_pred = []
        for step, (x, y) in enumerate(valid_loader):
            y = torch.squeeze(y)  # delete a axis
            if Args.cuda:
                x, y = x.cuda(), y.cuda()
            diagnosis.eval() # test model
            output = diagnosis(x)
            if Args.cuda:
                pred = torch.max(output, 1)[1].cuda().data.squeeze()
            else:
                pred = torch.max(output, 1)[1].data.squeeze()
            all_y.append(y)
            all_pred.append(pred)
        # evaluate
        y = torch.cat(all_y)
        pred = torch.cat(all_pred)
        accuracy = score.accuracy(pred, y)
        F1 = score.F1(pred, y)
        print('Epoch: %s | test accuracy: %.2f | F1: %.4f' % (epoch, accuracy, F1))
        # drawing
        if Args.draw:
            F1_list.append(F1)
            acc_list.append(accuracy)
            drawing.draw_result(acc_list, F1_list, figure, ['Accuracy', 'F1'], True)
        # save model
        if F1 == max(F1_list):
            print('save model')
            save_model = diagnosis.cpu()
            torch.save(save_model, '../transformer_models/model1.pkl')
            diagnosis = diagnosis.cuda()
            del save_model
        # empty memory
        del x, y, all_pred, all_y, output
        if Args.cuda: torch.cuda.empty_cache() # empty GPU memory
        # learning rate change
        # if epoch % 10 == 9:
        #     Args.learn_rate *= 0.9
        #     optimizer = torch.optim.Adam(diagnosis.parameters(), lr=Args.learn_rate) # optimizer
        #     print('changeing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('==>Finish')
    if Args.draw:
        plt.ioff()
        plt.show()








