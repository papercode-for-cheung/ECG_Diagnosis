# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/25 0:20
@File    : drawing.py
@Software: PyCharm
Introduction: Drawing the result
"""
#%% Import Packages
import matplotlib.pyplot as plt
import numpy as np
#%% Functions
def draw_result(result, fig, title=[], show=False):
    #  actionly draw the result
    num = len(result)
    # check
    if len(title) < num:
        for i in range(len(title), num):
            title.append(str(i))
    xaxis = [list(range(len(i))) for i in result] # axis -x
    subplot = []
    fig.clf()
    for i in range(num):
        subplot.append(fig.add_subplot(num, 1, i+1))
        subplot[i].plot(xaxis[i], result[i], marker='o')
        subplot[i].grid()
        subplot[i].set_title(title[i])
        if show:
            subplot[i].annotate(s=title[i] + ': %.3f' % result[i][-1], xy=(xaxis[i][-1], result[i][-1]),
                                xytext=(-20, 10), textcoords='offset points')
            r = np.array(result[i])
            subplot[i].annotate(s='Max: %.3f' % r.max(), xy=(r.argmax(), r.max()), xytext=(-20, -10),
                             textcoords='offset points')
    plt.pause(0.01)

#%% Main Function
if __name__ == '__main__':
    fig = plt.figure(1)
    plt.ion()
    b = []
    c = []
    d = []
    for i in range(100):
        a = np.random.randn(3)
        b.append(a[0])
        c.append(a[1])
        d.append(a[2])
        draw_result([b, c ,d], fig, ['b', 'c'], True)
    plt.ioff()
    plt.show()