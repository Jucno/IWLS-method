#!/usr/bin/env python
# coding: utf-8
# Author: Keyi Bin, Nanjing University

# # 自制波形测试

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from math import sin, cos, pi, radians, degrees, sqrt
from scipy.signal import convolve
from scipy.optimize import curve_fit
from random import choice
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import FuncFormatter

def gaussian_pulse(t, mu, sigma):
    """
    生成高斯脉冲信号

    参数：
    - t: 时间数组
    - mu: 高斯分布的均值
    - sigma: 高斯分布的标准差

    返回：
    - 高斯脉冲信号
    """
    return np.exp(-(t - mu)**2 / (2 * sigma**2))


# In[4]:


def plot_rt(t, n_values, R, T):
    """
    分别绘制R道和T道波形，并按方位角排列
    :param t: 时间轴
    :param n_values: 方位角数组
    :param R: R道波形
    :param T: T道波形    
    """
    # 创建画布和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    
    # 画图
    # 左边列画 f_n(t)
    for i, n in enumerate(n_values):
        ax1.plot(t, R[i] + n * 2, color='brown')
        # 在左边列标注 n 的值
        ax1.text(t[0]-1, R[i][0] + n * 2, str(round(degrees(n))), va='center', ha='right', color='brown')
    
    
    ax1.yaxis.set_visible(False)  # 隐藏左边的 y 轴
    ax1.spines[['right','top','left']].set_visible(False)  # 隐藏部分坐标轴边框
    
    # 右边列画 g_n(t)
    for i, n in enumerate(n_values):
        #放大了g的振幅
        ax2.plot(t, T[i] * 3 + n * 2, color='green')
    ax2.yaxis.set_visible(False)  # 隐藏左边的 y 轴
    ax2.spines[['right','top','left']].set_visible(False)  # 隐藏部分坐标轴边框
    
    # 在整张图的左右两侧设置 t 的刻度
    # plt.xticks([0, np.pi, 2*np.pi], ['0', '$\pi$', '$2\pi$'])
    
    # 调整布局
    plt.subplots_adjust(wspace=0.1)
    
    # 显示图形
    plt.show()

from pyraysum import prs, Model, Geometry, Control
import seispy
from seispy import decon

def get_prs_ray(model):
    """
    根据速度模型model获取相应波形。
    
    参数：
    model: 速度模型
    
    返回：
    prs.streamlist
    """

    # Define the ray geometry for one event
    geom = Geometry(np.arange(5., 360., 10.), 0.05)

    # Define the run-control parameters
    rc = Control(dt=0.01, npts=3000, mults=1, rot=1)
    # Run Raysum for this setup
    streamlist = prs.run(model, geom, rc, rf=False)
    streamlist.filter('streams', 'lowpass', freq=1., corners=2, zerophase=True)

    return streamlist


# In[14]:


def get_prs_rf(streamlist,f0):
    """
    从streamlist中提取波形并计算接收函数

    return:
    rfr, rft, t, n_values_radian
    """
    # 取出prs中的合成波形数据
    time_series = streamlist.streams[0][0].times()
    SYR = np.zeros((len(streamlist), streamlist.streams[0][0].stats.npts))
    SYT = np.zeros((len(streamlist), streamlist.streams[0][0].stats.npts))
    SYZ = np.zeros((len(streamlist), streamlist.streams[0][0].stats.npts))
    for i in range(len(streamlist.streams)):
        # print(type(rf[0]))
        SYR[i] = streamlist.streams[i][0].data
        SYT[i] = streamlist.streams[i][1].data
        SYZ[i] = streamlist.streams[i][2].data

    baz_degree = np.zeros(len(streamlist))
    for i, trace in enumerate(streamlist.streams):
        baz_degree[i] = trace[0].stats.baz

    #把角度转换成弧度
    baz_radian = baz_degree
    for i, baz in enumerate(baz_degree):
        baz_radian[i] = radians(baz)
        
    rfr = np.zeros_like(SYR)
    rft = np.zeros_like(SYT)
    shift = 10
    dt = 0.01
    itmax = 400
    minderr = 0.001
    for i in range(len(rfr)):
        rf1,_,_ = seispy.decon.deconit(SYR[i], SYZ[i], tshift=shift, f0=f0, itmax=itmax, minderr=minderr, dt = dt)
        rf2,_,_ = seispy.decon.deconit(SYT[i], SYZ[i], tshift=shift, f0=f0, itmax=itmax, minderr=minderr, dt = dt)
        rfr[i] = rf1
        rft[i] = rf2

    t = np.arange(-shift, -shift + len(rfr[0]) * dt, dt)

    return  t, baz_radian, rfr, rft



