# -*- coding: utf-8 -*-
"""
@Time    : 2019/3/31 17:36
@Author  : QuYue
@File    : arguments.py
@Software: PyCharm
Introduction: Get the input arguments.
"""
#%% Import Packages
import argparse
#%% Functions
def get():
    # Get the arguments from cmd.
    parser = argparse.ArgumentParser(description='ECG_Train')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('-n', '--datanum', type=int, default=600, metavar='N',
                        help='number of data for train (default: 600)')
    parser.add_argument('-r', '--train-ratio', type=float, default=0.8, metavar='Float',
                        help='ratio for splitting the train set from data set')
    parser.add_argument('-b', '--batch-size', type=int, default=20, metavar='N',
                        help='number of mini-batch size(default: 20)')
    parser.add_argument('-c', '--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('-d', '--draw', action='store_true', default=False,
                        help='enables dynamic draw the result')
    parser.add_argument('-l', '--learn-rate', type=float, default=0.001, metavar='Float',
                        help='learning-rate for training(default: 0.001)')
    parser.add_argument('-s', '--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    return args

def add(args, dictionary):
    # Add the others arguments to args.
    for i in dictionary.keys():
        exec("args.%s = dictionary[i]" %i)


#%% Main Function
if __name__ == '__main__':
    args = get()
    add(args, {'cuda': True})
    print("Cuda is %s" % args.cuda)
    print("ratio is %s" % args.train_ratio)