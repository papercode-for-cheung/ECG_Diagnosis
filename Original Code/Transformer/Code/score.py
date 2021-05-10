# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/1 21:04
@Author  : QuYue
@File    : score.py
@Software: PyCharm
Introduction: The score to evaluate the result by Tensor.
"""
#%% Import Packages
import torch
#%% Functions
def accuracy(pred, real):
    acc = float((pred == real).sum()) / float(real.size(0))
    return acc

def F1(pred, real):
    Aa = int((((pred == 1) + (real == 1)) == 2).sum())
    A = int((pred == 1).sum())
    a = int((real == 1).sum())
    if A + a == 0:
        f1 = 0
    else:
        f1 = 2 * Aa / (A+a)
    return f1
#%% Main Function
if __name__ == '__main__':
    pred = torch.Tensor([0, 0, 0, 1, 1])
    real = torch.Tensor([1, 0, 0 ,1 ,0])
    acc = accuracy(pred, real)
    f1 = F1(pred, real)
    print('accuracy: %.2f, f1: %.4f' %(acc, f1))