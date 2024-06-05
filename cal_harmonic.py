import matplotlib
import numpy as np
from math import cos, sin, pi
import os
import obspy
import matplotlib.pyplot as plt
from matplotlib import patches
import seispy
from seispy import decon

from seispy.rfcorrect import SACStation, moveoutcorrect_ref
from os.path import join
matplotlib.use('TkAgg')

def plot_motion(EW, NS):
    plt.plot(EW, NS)
    plt.show()


def read_finallsit(data_path):
    f1 = open(data_path)
    f1_lines = f1.readlines()
    evnum = len(f1_lines)
    R_file = []
    T_file = []
    for line in f1_lines:
        line = line.strip()
        line = line.split(' ')
        R_file.append(line[0] + '_P_R.sac')
        T_file.append(line[0] + '_P_T.sac')
    return R_file, T_file, evnum

def readchannel(filename):
    file = open(filename)
    file_lines = file.readlines()
    pts = len(file_lines)
    data = []
    for line in file_lines:
        line = line.strip()
        formline = line.split(' ')
        data.append(float(formline[0]))
    return pts, data

def cal_syn():
    pt = 1000
    rfr = np.zeros([36,pt])
    rft = np.zeros([36,pt])
    for i in range(36):
        R_path = '/Users/jicong/Downloads/Raysum_fwd_v1.2/bin/R' + str(i+1) + '.tr'
        T_path = '/Users/jicong/Downloads/Raysum_fwd_v1.2/bin/T' + str(i+1) + '.tr'
        Z_path = '/Users/jicong/Downloads/Raysum_fwd_v1.2/bin/Z' + str(i+1) + '.tr'
        _, data1 = readchannel(R_path)
        _, data2 = readchannel(T_path)
        _, data3 = readchannel(Z_path)
        shift = 2
        dt = 0.01
        f0 = 2
        itmax = 100
        minderr = 0.001
        rf,_,_ = seispy.decon.deconit(data1, data3, tshift=shift, f0=f0, itmax=itmax, minderr=minderr, dt = dt)
        rt,_,_ = seispy.decon.deconit(data2, data3, tshift=shift, f0=f0, itmax=itmax, minderr=minderr, dt = dt)
        rfr[i] = rf[:1000]
        rft[i] = rt[:1000]
    baz = np.arange(5,360,10)
    sac_data = np.concatenate([rfr, rft])
    ave_Rdata = np.sum(sac_data[:(36-1), :], axis=0) / 36
    ave_Tdata = np.sum(sac_data[36:, :], axis=0) / 36
    return sac_data, baz * pi / 180, ave_Rdata, ave_Tdata

def readsac(data_path, station):
    sacdata = SACStation(join(data_path, station, station+'finallist.dat'))
    Rsac = sacdata.datar
    try:
        R_file, T_file, evnum = read_finallsit(os.path.join(
            data_path, station, station + 'finallist.dat'))
    except:
        R_file, T_file, evnum = read_finallsit(
            os.path.join(data_path, station, 'finallist.dat'))
    sac_data = np.zeros((2 * evnum, 2000))#1000
    ave_Rdata = np.zeros(evnum)
    ave_Tdata = np.zeros(evnum)
    baz1 = np.zeros(evnum)
    # 叠加步长
    phi_step = 10
    for i in range(evnum):
        if (i == 0):
            R_sac = obspy.read(os.path.join(data_path, station, R_file[i]))
            T_sac = obspy.read(os.path.join(data_path, station, T_file[i]))
        else:
            R_sac += obspy.read(os.path.join(data_path, station, R_file[i]))
            T_sac += obspy.read(os.path.join(data_path, station, T_file[i]))
        sac_data[i, :] = R_sac[i].data[0:2000]#[800:1800]
        sac_data[i + evnum, :] = T_sac[i].data[0:2000]
        baz1[i] = R_sac[i].stats.sac.baz
    ave_Rdata = np.sum(sac_data[:(evnum-1), :], axis=0)
    ave_Tdata = np.sum(sac_data[evnum:, :], axis=0)
    R_stack = np.zeros((int(360 / phi_step), 2000))
    T_stack = np.zeros((int(360 / phi_step), 2000))
    baz = np.arange(phi_step / 2, 360, phi_step)
    for j in range(0, 360, phi_step):
        count = 0
        for i in range(evnum):
            if baz1[i] > j and baz1[i] < j + phi_step:
                R_stack[int(j/phi_step)] += sac_data[i]
                T_stack[int(j/phi_step)] += sac_data[i + evnum]
                count += 1
        if count > 0:
            R_stack[int(j/phi_step)] /= count
            T_stack[int(j/phi_step)] /= count
    loc1 = np.where(~R_stack.any(axis=1))[0]
    R_stack = np.delete(R_stack, loc1, 0)
    T_stack = np.delete(T_stack, loc1, 0)
    baz = np.delete(baz, loc1, 0)
    sac_data1 = np.concatenate([R_stack, T_stack])
    return sac_data1, baz * pi / 180, ave_Rdata / evnum, ave_Tdata / evnum


def cal_harmonic(station):
    if sta_lst != ['synthetic']:
        data_path = '/Users/jicong/Desktop/RFresult/'
        sac_data, baz, ave_R, ave_T = readsac(data_path, station)
    else:
        sac_data, baz, ave_R, ave_T = cal_syn()

    coef = np.zeros((sac_data.shape[0], 5))
    coef2 = np.zeros((sac_data.shape[0], 5))
    half_shape = int(sac_data.shape[0] / 2)
    result = np.zeros((5, 1000))
    result2 = np.zeros((5, 1000))
    for i in range(half_shape):
        coef[i, 0] = 1
        coef[i + half_shape, 0] = 0
        coef[i, 1] = cos(baz[i])
        coef[i + half_shape, 1] = cos(baz[i] + pi / 2)
        coef[i, 2] = sin(baz[i])
        coef[i + half_shape, 2] = sin(baz[i] + pi / 2)
        coef[i, 3] = cos(2 * baz[i])
        coef[i + half_shape, 3] = cos(2 * baz[i] + pi / 2)
        coef[i, 4] = sin(2 * baz[i])
        coef[i + half_shape, 4] = sin(2 * baz[i] + pi / 2)
    # for i in range(half_shape):
        coef2[i, 0] = 0
        coef2[i + half_shape, 0] = 1
        coef2[i, 1] = cos(baz[i])
        coef2[i + half_shape, 1] = cos(baz[i] - pi / 2)
        coef2[i, 2] = sin(baz[i])
        coef2[i + half_shape, 2] = sin(baz[i] - pi / 2)
        coef2[i, 3] = cos(2 * baz[i])
        coef2[i + half_shape, 3] = cos(2 * baz[i] - pi / 2)
        coef2[i, 4] = sin(2 * baz[i])
        coef2[i + half_shape, 4] = sin(2 * baz[i] - pi / 2)
    for i in range(1000):
        sac_t = np.mat(sac_data[:, i]).T
        # x = np.linalg.lstsq(np.array(coef), sac_t, rcond=None)
        left = np.dot(np.mat(coef).T, sac_t)
        right = np.dot(np.mat(coef).T, np.mat(coef))
        x = np.dot(right.I, left)
        for j in range(5):
            result[j, i] = x[j]  # x[0].item(j)
        # x2 = np.linalg.lstsq(np.array(coef2), sac_t, rcond=None)
        left = np.dot(np.mat(coef2).T, sac_t)
        right = np.dot(np.mat(coef2).T, np.mat(coef2))
        x2 = np.dot(right.I, left)
        for j in range(5):
            result2[j, i] = x2[j]  # x2[0].item(j)
    return result, result2, baz, ave_R, ave_T


if __name__ == '__main__':
    data_path = '/Users/jicong/Desktop/RFresult/'
    # sta_lst_dir = os.path.join(data_path, 'shanxi/sta.lst')
    # sta_lst = np.genfromtxt(sta_lst_dir, dtype='str')
    # nbaz = '1'+'.tr'
    # sta_lst = ['14884','14891','14896','14895','14879','14867','14892','14877','14852']
    sta_lst = ['synthetic']
    x = np.linspace(-2, 8, 1000)

    for station in sta_lst:
        result, result2, baz, ave_R, ave_T = cal_harmonic(station)
        # plt.plot(result[3, 400:800], result[4, 400:800])
        # plt.show()
        # result /= 2
        # result2 /= 2
        result *=4
        result2 *=4
        result[0] /= 5
        result2[0] /= 5
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.style.use('bmh')
        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (14,5), tight_layout = False)
        ax[0].set_yticks([0, -0.1, -0.2, -0.3, -0.4])
        ax[0].set_ylim([-0.5, 0.15])
        ax[0].set_yticklabels(['   const','  cos($\phi$)','  sin($\phi$)',' cos(2$\phi$)',' sin(2$\phi$)'], fontsize = 16)
        ax[1].set_yticks([0, -0.1, -0.2, -0.3, -0.4])
        ax[1].set_ylim([-0.5, 0.15])
        ax[1].set_yticklabels([])
        ax[0].set_xlabel('Delay time(s)', fontsize = 16)
        ax[1].set_xlabel('Delay time(s)', fontsize = 16)
        ax[0].yaxis.set_ticks_position('right')
        ax[0].set_title('Dip/Anisotropy', fontsize = 20)
        ax[1].set_title('Unmodeled', fontsize = 20)

        for i in range(result.shape[0]):
            ax[0].plot(x,result[i]-i/10, color='black', linewidth = 0.05, alpha = .4)
            ax[0].fill_between(x, -i/10, result[i]-i/10, result[i]>0, color = 'red', alpha = .32)
            ax[0].fill_between(x, -i/10, result[i]-i/10, result[i]<0, color = 'blue', alpha = .32)
        for i in range(result.shape[0]):
            ax[1].plot(x,result2[i]-i/10, color='black', linewidth = 0.05, alpha = .4)
            ax[1].fill_between(x, -i/10, result2[i]-i/10, result2[i]>0, color = 'red', alpha = .32)
            ax[1].fill_between(x, -i/10, result2[i]-i/10, result2[i]<0, color = 'blue', alpha = .32)
        #rect = patches.Rectangle((1,-0.5),1,0.6,fill=True, linewidth=0.1,facecolor = 'green', alpha=0.2)
        #rect1 = patches.Rectangle((2.5,-0.5),1.5,0.6,fill=True, linewidth=0.1,facecolor = 'orange', alpha=0.2)
        #ax[0].add_patch(rect)
        #ax[0].add_patch(rect1)
        filename = station + '.png'
        # plt.suptitle('Station  ' + station, fontsize = 24, x = 0.51)
        plt.savefig('paperfig/' + filename, format = 'png', dpi = 600)
        plt.show()
        plt.clf()
        print(station)
    plt.close()

    # sta_lst = ['13807']
    # for new_sta in sta_lst:
    #     TR, nbaz, aR, aT = readsac(data_path, new_sta)
    #     Tnum = int(TR.shape[0] / 2)
    #     Awst = np.zeros((18,1000))
    #     for psi in range(0,180,10):
    #         sum = 0
    #         for evNumber in range(Tnum):
    #             sum += sin( 2 * ( psi * pi / 180 - nbaz[evNumber] ) ) * sin( 2 * ( psi * pi / 180 - nbaz[evNumber] ) )
    #             # Awst[int(psi / 10)] += -sin( 2 * ( psi * pi / 180 - nbaz[evNumber] ) ) * TR[Tnum + evNumber]
    #         for evNumber in range(Tnum):
    #             Awst[int(psi / 10)] += sin( 2 * ( psi * pi / 180 - nbaz[evNumber] ) ) / sum * TR[Tnum + evNumber]

    #     whereMax = 120
    #     t0 = 3
    #     t1 = 5
    #     t0 += 2
    #     t1 += 2
    #     maxLoc = np.argmax(Awst[int(whereMax / 10), t0*100:t1*100])
    #     max_t = maxLoc / 100 + t0 - 2

    #     newbaz = nbaz * 180 / pi - 2.5
    #     yLabel = np.zeros(18)
    #     for i in range(18):
    #         yLabel[i] = 10 * i 
    #     plt.figure(figsize=(8,15))
    #     plt.style.use('bmh')
    #     plt.rcParams['font.family'] = 'Times New Roman'
    #     plt.title('Station ' + new_sta, fontsize = 35)
    #     plt.xlabel('Delay time (s)', fontsize = 28)
    #     plt.ylabel('Back azimuth ($^\circ$)',fontsize = 28)
    #     plt.yticks(yLabel, fontsize = 22)
    #     plt.xticks(fontsize = 22)
    #     amp = 200
    #     TR *= 1.6
    #     for i in range(yLabel.shape[0]):
    #         plt.plot(x, amp * Awst[i] + yLabel[i], color='black', linewidth = 0.6, alpha = .8)
    #         plt.fill_between(x, yLabel[i], amp * Awst[i] + yLabel[i], Awst[i] > 0, color = 'red', alpha = .4)
    #         plt.fill_between(x, yLabel[i], amp * Awst[i] + yLabel[i], Awst[i] < 0, color = 'blue', alpha = .4)
    #     plt.plot(max_t, whereMax, color = '#fff917', marker = '+', linestyle = '-', markersize = 40, linewidth = 50, markeredgewidth = 7)
    #     plt.savefig('AwstForpaper/' + new_sta)
    #     print(new_sta)