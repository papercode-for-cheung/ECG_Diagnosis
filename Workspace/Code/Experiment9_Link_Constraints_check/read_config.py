# -*- coding: utf-8 -*-
"""
@Time    : 2019/5/27 21:58
@File    : read_config.py
@Software: PyCharm
Introduction: Read the configure file.
"""
#%% Import Packages
import configparser

#%% Functions
def read(Args):
    # Read
    cf = configparser.ConfigParser()
    cf.read(Args.config)
    # Add [data_read]
    Args.datanum = cf.getint('data_read', 'datanum')
    Args.data_path = cf.get('data_read', 'data_path')
    Args.label_path = cf.get('data_read', 'label_path')
    # Add [data_process]
    Args.trainratio = cf.getfloat('data_process', 'trainratio')
    Args.win_size = cf.getint('data_process', 'win_size')
    Args.win_step = cf.getint('data_process', 'win_step')
    # Add [train]
    Args.batch_size = cf.getint('train', 'batch_size')
    Args.epoch = cf.getint('train', 'epoch')
    Args.learn_rate = cf.getfloat('train', 'learn_rate')
    Args.cuda = cf.getboolean('train', 'cuda')
    Args.link_weight = cf.getfloat('train', 'link_weight')
    # Add [show]
    Args.show_plot = cf.getboolean('show', 'show_plot')
    Args.show_log = cf.getboolean('show', 'show_log')
    return Args



#%% Main Function
if __name__ == '__main__':
    class Argument():
        def __init__(self):
            self.config = './Config/config.ini'
    Args = Argument()
    Args = read(Args)
    show = Args.link_weight
    print(show, type(show))

