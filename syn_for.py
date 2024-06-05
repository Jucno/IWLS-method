import seispy
from cal_harmonic import *
from pyraysum import prs, Model, Geometry, Control
import importlib
import numpy as np
from math import sin, cos, pi, sqrt, atan2
from scipy.sparse.linalg import lsqr
from scipy.signal import find_peaks
from scipy import interpolate
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from seispy.geo import skm2srad
from seispy.rfcorrect import SACStation, moveoutcorrect_ref
from seispy.rf2depth_makedata import Station
from seispy import decon
from numpy import matrix
from math import exp, tan
import matplotlib
matplotlib.use('TkAgg')
mySImodule = importlib.import_module("SI_try")

def xcorr(a,b,mode ='coef'):
    fa = np.fft.fft(a,n=len(a)*2-1)
    fb = np.fft.fft(b,n=len(a)*2-1)
    xx = fa*np.conj(fb)
    xcc = np.fft.fftshift(np.fft.ifft(xx))
    aa = fa*np.conj(fa)
    xaa = np.fft.fftshift(np.fft.ifft(aa))
    bb = fb*np.conj(fb)
    xbb = np.fft.fftshift(np.fft.ifft(bb))
    if mode == 'coef':
        xcc = xcc/np.sqrt(xaa[len(a)-1]*xbb[len(b)-1])
    return np.real(xcc)

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
    rfr = np.zeros([36,pt])
    # for i in range(36):
    #     R_path = '/Users/jicong/Downloads/Raysum_fwd_v1.2/bin/R' + str(i+1) + '.tr'
    #     T_path = '/Users/jicong/Downloads/Raysum_fwd_v1.2/bin/T' + str(i+1) + '.tr'
    #     Z_path = '/Users/jicong/Downloads/Raysum_fwd_v1.2/bin/Z' + str(i+1) + '.tr'
    #     _, data1 = readchannel(R_path)
    #     _, data3 = readchannel(Z_path)
    #     shift = 10
    #     dt = 0.01
    #     f0 = 2
    #     itmax = 100
    #     minderr = 0.001
    #     rf,_,_ = seispy.decon.deconit(data1, data3, tshift=shift, f0=f0, itmax=itmax, minderr=minderr, dt = dt, nt=pt)
    #     rfr[i] = rf

    #     else:
    #         rfr_c = rfr.copy()
    #         baz = np.arange(5,360,10)
    baz = np.arange(5,360,10)
    thick = [35000, 5000.,0]
    # rho = [2900., 3190,3600.]
    rho = [2900., 2900,3600.]
    # vp = [6500., 7150,8040.]
    # vs = [3750., 4125,4500.]
    vp = [6500., 6500,8040.]
    vs = [3750., 3750,4500.]
    flag = [1, 0,1]#[0,1]
    ani = [0, 20,0]
    trend = [0., 0.,0]
    plunge = [0., 0.,0]
    strike = [0,0,0]
    dip = [0, 0,0]
    # vpvs = [1.75,1.75]

    model = Model(thick, rho, vp, vs, flag = flag,ani=ani, trend=trend, plunge=plunge, strike=strike, dip=dip)
    # model.plot()

    streamlist = mySImodule.get_prs_ray(model)
    print(streamlist)
    _, _, rfr, _ = mySImodule.get_prs_rf(streamlist,6)
    if is_boot:
        total_num = 18
        rfr_c = np.zeros([total_num,pt])
        num = np.arange(0,36,1)
        baz_raw = np.arange(5,360,10)

        baz_count = np.random.choice(num, total_num, replace=False)
        len_baz = len(baz_count)
        baz = np.zeros(len_baz)
        for i in range(len_baz):
            baz[i] = baz_raw[int(baz_count[i])]
            rfr_c[i] = rfr[int(baz_count[i])]
    else:
        rfr_c = rfr.copy()
        baz = np.arange(5,360,10)
    return rfr_c, baz
    


def read_data(station, path='./', ref_rayp=None):
    sac_data = SACStation(join(path, station, station+'finallist.dat'))
    if isinstance(ref_rayp, float):
        sac_data.datar, _ = moveoutcorrect_ref(sac_data, skm2srad(ref_rayp), np.arange(100), sac_data.sampling, sac_data.shift)
    time_axis = np.arange(0, sac_data.RFlength*sac_data.sampling, sac_data.sampling) - sac_data.shift
    return sac_data, time_axis


def stack_baz(sac_data, val=10):
    stack_range = np.arange(0, 360+val, val)
    rft_baz = np.zeros([stack_range.shape[0]-1, sac_data.RFlength])
    rfr_baz = np.zeros_like(rft_baz)
    count_baz = np.zeros(stack_range.shape[0]-1)
    for i in range(stack_range.shape[0]-1):
        idx = np.where((sac_data.bazi > stack_range[i]) & (sac_data.bazi < stack_range[i+1]))[0]
        count_baz[i] = idx.size
        if idx.size != 0:
            rft_baz[i] = np.mean(sac_data.datat[idx], axis=0)
            rfr_baz[i] = np.mean(sac_data.datar[idx], axis=0)
    return rfr_baz, rft_baz, count_baz

def plotR(floc, sloc, omega, s, xmax, ymax,dt, phi, deltat, t0, A1, A2, phi_2phi, phi_1phi, rfr_baz, bazs, time_axis, val=10, enf=100, xlim=[-1, 20]):
    enf = 100
    ml = MultipleLocator(5)
    #bazs = np.arange(0, 360, val)+val/2
    bound = np.zeros_like(time_axis)
    plt.style.use("bmh")
    
    plt.figure(figsize=(12, 12))
    plt.title("111")
    # axr = plt.axes()
    grid = plt.GridSpec(5,5)
    axr = plt.subplot(grid[1:,0:3])
    axr2 = plt.subplot(grid[1:,3])
    axr3 = plt.subplot(grid[1:,4])
    ax0 = plt.subplot(grid[0,:])
    ax0.axis('off')
    ax0.set_xlim((-2,2))
    ax0.set_ylim((-3,3))
    ax0.plot([-0.75,0.75],[2.8,2.8])
    ax0.plot([-0.75,0.75],[-0.425,-0.425])
    case_name = '34'
    inclination = 0
    y_moho = 0.75 * tan(inclination * pi / 180)
    ax0.plot([-0.75,0.75], [-1.5 - y_moho, -1.5+y_moho], color='C0')
    # ax0.text(0,0.1,'5'+'%' + ' anisotropy, FPD 0$^\circ$\nMoho dips {:}$^\circ$ to 90$^\circ$'.format(inclination), horizontalalignment='center', fontsize = 18)
    # ax0.text(0,0.1,'Isotropic crust\nMoho dips 15$^\circ$ to 90$^\circ$',horizontalalignment='center', fontsize = 18)
    ax0.text(0,0.9,'Isotropic',horizontalalignment='center', fontsize = 18)
    ax0.text(0,-1.12,'20'+'%' + ' anisotropy',horizontalalignment='center', fontsize = 14)
    # ax0.text(0,0.1,'5'+'%' + ' anisotropy, FPD 0$^\circ$\nFlat Moho', horizontalalignment='center', fontsize = 18)
    ax0.set_title('Case ({:})'.format(case_name), fontsize = 22)
    plt.subplots_adjust(hspace=None)
    axr2.set_yticks(())
    axr3.set_yticks(())
    
    for i in range(bazs.shape[0]):
        if np.mean(rfr_baz[i]) == 0:
            continue
        amp =1.2 * rfr_baz[i] * enf + bazs[i]
        axr.fill_between(time_axis, amp, bound + bazs[i], where=amp > bazs[i], facecolor='red', alpha=0.7)
        axr.fill_between(time_axis, amp, bound + bazs[i], where=amp < bazs[i], facecolor='#1193F4', alpha=0.7)
    if is_WLS:
        for i in range(bazs.shape[0]):
            ta = np.arange(floc[i] / 100 - 10, sloc[i] / 100 - 10, 0.001)
            axr2.plot(ta,1.2 * enf * ymax[i] * np.exp (-(ta - xmax[i])**2 / s[i]) + bazs[i], color='green')
    axr.set_xlim(xlim)
    axr.xaxis.set_major_locator(MultipleLocator(1))
    axr.set_ylim(0, 365)
    axr2.set_ylim(0, 365)
    axr3.set_ylim(0, 365)
    axr.set_yticks(np.arange(0, 360+30, 30))
    axr.yaxis.set_minor_locator(ml)
    axr.set_ylabel('Back-azimuth ($^\circ$)',fontsize=20)
    axr.set_xlabel('Time after P (s)',fontsize=20)
    axr2.set_xlabel('Time after P (s)',fontsize=20)
    # axr.set_title('R component ({})'.format(station))
    # axr.set_title('R component (Case '+case_name+')')
    axr.set_title('R component',fontsize=20)
    axr2.set_title('Fitted curves',fontsize=20)
    axr3.set_title('Weights',fontsize=20)

    if A2 < 0.035:
        A2 = 0
        phi_1phi = 0
    if A1 < 0.035:
        A1 = 0
        phi_2phi = 0
    if phi_2phi > 90:
        phi_2phi -=180
    for i in [0, 45, 90, 135]:
        if abs(phi_1phi - i)<1.5:
            phi_1phi = i
            continue
    for i in [0, 45, 30, 90, 135]:
        if abs(phi_2phi - i)<1.5:
            phi_2phi = i
            continue
    # phi_1phi = 48
    # phi_2phi = -1
    # if phi_1phi >180 :
    #     phi_1phi += 180

    for i in range(len(dt)):
        axr.scatter(dt[i], bazs[i], marker='x', color = 'k')
        axr3.scatter(omega[i], bazs[i], marker='o', color = 'k')
    x = np.arange(0,360,0.1)
    y = np.arange(0,360,0.1)
    y2 = np.arange(0,360,0.1)
    for i in range(x.shape[0]):
        y[i] = t0 - A1 * cos(2 * (phi_2phi - x[i]) * pi / 180) + A2 * cos((phi_1phi - x[i]) * pi / 180)
        if count > -1:
            y2[i] = t0 - deltat / 2 * cos(2 * (phi * pi / 180 - x[i] * pi / 180))
    axr.plot(y,x,color='black')
    # A1 = 0.28
    # phi_1phi = 90
    # A2 = 0.27
    # phi_2phi = 0

    axr.text(1.15,320,'A1=%.2fs   $\phi$1=%d$^\circ$\nA2=%.2fs   $\phi$2=%d$^\circ$'%(A2, round(phi_1phi), A1, round(phi_2phi)), fontsize = 16)
    print('WLS: {:.4f} {:.4f} {:.4f} {:.4f}'.format(A2, phi_1phi, A1, phi_2phi))
    # print('\n')
    # plt.savefig('/Users/jicong/Desktop/RFresult/shanxi/anistropy/lsqr_for_new/case'+case_name+'.png',format='png', dpi=400, bbox_inches='tight')
    # plt.savefig('/Users/jicong/Desktop/RFresult/shanxi/anistropy/lsqr_revision/case'+case_name+'.png',format='png', dpi=400, bbox_inches='tight')
    # plt.savefig('/Users/jicong/Desktop/RFresult/shanxi/anistropy/lsqr_for_new/figure_9.png',format='png', dpi=400, bbox_inches='tight')
    plt.savefig('/Users/jicong/Desktop/high.png',format='jpg', dpi=400, bbox_inches='tight')
    # if count > -1:
    #     plt.plot(y2,x,color='orange')
    #     plt.savefig('/Users/jicong/Desktop/RFresult/shanxi/anistropy/lsqr_test/{:}.png'.format(station), format='png', dpi=800, bbox_inches='tight')
    # else:
    #     plt.savefig('/Users/jicong/Desktop/RFresult/shanxi/anistropy/lsqr_for/model_test3.png',format='png', dpi=800, bbox_inches='tight')
    if is_showFigure:
        plt.show()


#------Examine if the picked dts are evidentally wrong and repick dts if so------#
def examine_dt(start, end, dt, Rsac, baz, thresh):
    time_step = 25
    phi_num = len(dt)
    Rvalue = np.zeros(phi_num)
    start_new = start
    end_new = end
    for i in range(phi_num):
        Rvalue[i] = Rsac[i, int(1000 + 100 * dt[i])]

    #------Determine if a negative one is picked (at the start of the selected time window)------#
    dt_lim = 4 #The max iteration number
    dt_iter = 0
    # while np.any(Rvalue<0) and (dt.min() - (start - 1000) * 0.01) > 0.02 and dt_iter < dt_lim:
    while np.any(Rvalue<0) and dt_iter < dt_lim:
        nega_loc = np.where(Rvalue<0)[0]
        nega_num = len(nega_loc)
        if nega_num > 1:
            Rsac_nega = np.zeros([nega_num, pt])
            for i in range(nega_num):
                Rsac_nega[i, :] = Rsac[nega_loc[i], :]
            start_new -= time_step
            end_new += time_step
            dt_new = pickdt(start_new, end_new, Rsac_nega, 0) 
            count_dt = 0
            for i in range(nega_num):
                dt[nega_loc[i]] = dt_new[count_dt]
                count_dt += 1
            for i in range(phi_num):
                Rvalue[i] = Rsac[i, int(1000 + 100 * dt[i])]
            dt_iter += 1
            if dt_iter == dt_lim:
                nega_loc = np.where(Rvalue<0)[0]
                dt = np.delete(dt, nega_loc, 0)
                baz = np.delete(baz, nega_loc, 0)
                Rsac = np.delete(Rsac, nega_loc, 0)
                Rvalue = np.delete(Rvalue, nega_loc, 0)
                #print('{}: Cannot find the PmS peaks of some bazs, please check the stacked RF results!'.format(station))
                wrong_stalst.append(station)
        else:
            break
    iter2_num = 0
    iter2_lim = 4
    #while (dt.min() * 100 + 1000 - start) < 2 or (dt.max() * 100 + 1000 - end) > -2:
    wrongmin_loc = np.where(dt - (start - 1000) * 0.01 < 0.02)[0]
    wrongmax_loc = np.where(dt - (end - 1000) * 0.01 > -0.02)[0]
    min_len = len(wrongmin_loc)
    max_len = len(wrongmax_loc)
    if min_len>0:
        start_new = start
        Rsac_min = np.zeros([min_len, pt])
        try:
            for i in range(min_len):
                Rsac_min[i, :] = Rsac[wrongmin_loc[i], :]
        except:
            Rsac_min = Rsac[wrongmin_loc[i], :]
        while iter2_num < iter2_lim and (dt.min() * 100 + 1000 - start_new) < 3:
            start_new -= time_step
            dt_new = pickdt(start_new, end, Rsac_min, 0)
            iter2_num += 1
        for i in range(min_len):
            dt[wrongmin_loc[i]] = dt_new[i]
    iter2_num = 0
    if max_len>0:
        end_new = end
        Rsac_max = np.zeros([max_len, pt])
        try:
            for i in range(max_len):
                Rsac_max[i, :] = Rsac[wrongmax_loc[i], :]
        except:
            Rsac_max = Rsac[wrongmax_loc[i], :]
        while iter2_num < iter2_lim and (dt.max() * 100 + 1000 - end_new) > -3:
            end_new -= time_step
            dt_new = pickdt(start, end_new, Rsac_max, 0)
            iter2_num += 1
        for i in range(max_len):
            dt[wrongmax_loc[i]] = dt_new[i]

    phi_num = len(dt)
    dt_mean = dt.mean()

    if dt_iter < dt_lim:
        sampling_times = 10000
        sampling_number = 5
        for ss in range(sampling_times):
            dt_sample = np.random.choice(dt, sampling_number, replace = True)
            if abs(dt_sample.mean() - dt_mean) > thresh:
                #print('{}: Confusing range of PmS arrival time!'.format(station))
                lisan_stalst.append(station)
                for i in range(phi_num):
                    iter_num = 0
                    while abs(dt[i] - (dt_mean * phi_num - dt[i]) / (phi_num - 1)) > thresh and iter_num < 3:
                        dt_corr = pickdt(start - 2 * time_step, end + 2 * time_step, Rsac[i, :], iter_num)
                        iter_num += 1
                        dt[i] = dt_corr[0]
                break

    return dt, baz
            

def cal_ani(count):
    # start = 1300
    # end = 1600
    diff_max = 0.8
    diff_iter_lim = 2
    # if is_syn == 0:
    #     sac_data, time_axis = read_data(station, path=data_path, ref_rayp=0.06)
    #     rfr_baz, _, _ = stack_baz(sac_data, val=val)
    #     baz = np.arange(0, 360, val) + val / 2
    # else:
    it =  1
    A1_ = np.zeros(it)    
    A2_ = np.zeros(it)
    phi_1phi_ = np.zeros(it)
    phi_2phi_ = np.zeros(it)
    nn = 100
    for tt in range(it):
        rfr_baz, baz = cal_syn()
        if is_noise:
            for i in range(rfr_baz.shape[0]):
                noise = np.random.normal(0.004,0.02,nn)
                noise1 = np.random.normal(0.004,0.02,nn)
                # noise2 = np.random.normal(0.008,0.015,nn)
                time_axis = np.arange(-10,20,0.01)
                nx = np.linspace(-10,20,nn)
                f = interpolate.interp1d(nx, noise, kind='cubic')
                f1 = interpolate.interp1d(nx, noise1, kind='cubic')
                # f2 = interpolate.interp1d(nx, noise2, kind='cubic')
                # noise_new2 = f2(time_axis)
                noise_new = f(time_axis)
                noise_new1 = f1(time_axis)
                rfr_baz[i] = rfr_baz[i] + noise_new 
        #baz = np.arange(0, 360, val) + val / 2
        loc1 = np.where(~rfr_baz.any(axis=1))[0]
        rfr_baz = np.delete(rfr_baz, loc1, 0)
        baz = np.delete(baz, loc1, 0)

        # for rrr in range(4):
        #     rfr_baz = np.delete(rfr_baz, 25, 0)
        #     baz = np.delete(baz, 25, 0)

        dt = pickdt(start, end, rfr_baz, 0)
        dt, baz = examine_dt(start, end, dt, rfr_baz, baz, thresh)
        phi_m = np.zeros([baz.shape[0], 5])
        phi_m[:, 0] = 1
        for i in range(baz.shape[0]):
            phi_m[i, 1] = cos(2 * baz[i] * pi / 180)
            phi_m[i, 2] = sin(2 * baz[i] * pi / 180)
            phi_m[i, 3] = cos(baz[i] * pi / 180)
            phi_m[i, 4] = sin(baz[i] * pi / 180)

        m2 = np.array(list(lsqr(phi_m,dt,iter_lim=50,damp=0,atol=1e-10,btol=1e-10)[:1])).T
        A11 = sqrt(m2[1]**2 + m2[2]**2)
        A21 = sqrt(m2[3]**2 + m2[4]**2)
        phi_2phi1 = 0.5 * atan2(-m2[2], -m2[1]) * 180 / pi
        phi_1phi1 = atan2(m2[4], m2[3]) * 180 / pi


        if is_WLS:
            dt_tmp = dt.copy()
            omega, s, xmax, ymax, floc, sloc = cal_omega_mat(rfr_baz, dt)
            for i in range(phi_m.shape[0]):
                phi_m[i] *= omega[i] 
                dt_tmp[i] *= omega[i]
            m = np.array(list(lsqr(phi_m,dt_tmp,iter_lim=50,damp=0)[:1])).T
        else:
            m = np.array(list(lsqr(phi_m,dt,iter_lim=50,damp=0,atol=1e-10,btol=1e-10)[:1])).T


        A1 = sqrt(m[1]**2 + m[2]**2)
        A2 = sqrt(m[3]**2 + m[4]**2)
        phi_2phi = 0.5 * atan2(-m[2], -m[1]) * 180 / pi
        phi_1phi = atan2(m[4], m[3]) * 180 / pi
        diff_iter_num = 0
        while diff_iter_num < diff_iter_lim:
            dt_cal = np.zeros(len(baz))
            for i in range(len(dt_cal)):
                dt_cal[i] = m[0] - A1 * cos(2 * (phi_2phi - baz[i]) * pi / 180) + A2 * cos((phi_1phi - baz[i]) * pi / 180)

            #------Delete the certain dts if they are not close enough to the theoretical values------#
            dt_diff = abs(dt_cal - dt)
            diff_loc = np.where(dt_diff > diff_max)[0]
            if len(diff_loc) > 2000:
                dt = np.delete(dt, diff_loc, 0)
                baz = np.delete(baz, diff_loc, 0)
                rfr_baz = np.delete(rfr_baz, diff_loc, 0)
                dt = pickdt(start, end, rfr_baz, 0)
                dt, baz = examine_dt(start, end, dt, rfr_baz, baz, thresh)
                phi_m = np.zeros([baz.shape[0], 5])
                phi_m[:, 0] = 1
                for i in range(baz.shape[0]):
                    phi_m[i, 1] = cos(2 * baz[i] * pi / 180)
                    phi_m[i, 2] = sin(2 * baz[i] * pi / 180)
                    phi_m[i, 3] = cos(baz[i] * pi / 180)
                    phi_m[i, 4] = sin(baz[i] * pi / 180)
                if is_WLS:
                    dt_tmp = dt.copy()
                    omega, s, xmax, ymax, floc, sloc = cal_omega_mat(rfr_baz, dt)
                    for i in range(phi_m.shape[0]):
                        phi_m[i] *= omega[i] 
                        dt_tmp[i] *= omega[i]
                        m = np.array(list(lsqr(phi_m,dt_tmp,iter_lim=50,damp=0)[:1])).T
                else:
                    m = np.array(list(lsqr(phi_m,dt,iter_lim=50,damp=0)[:1])).T
                A1 = sqrt(m[1]**2 + m[2]**2)
                A2 = sqrt(m[3]**2 + m[4]**2)
                phi_2phi = 0.5 * atan2(-m[2], -m[1]) * 180 / pi
                phi_1phi = atan2(m[4], m[3]) * 180 / pi
                diff_iter_num += 1
            else:
                diff_iter_num = diff_iter_lim
        A1_[tt] = A2
        A2_[tt] = A1
        phi_1phi_[tt] = phi_1phi
        phi_2phi_[tt] = phi_2phi    
        if A2 < 0.01:
            A2 = 0
            phi_1phi = 0
        if A1 < 0.01:
            A1 = 0
            phi_2phi = 0
        if phi_2phi < -0.5:
            phi_2phi += 180

        if A21 < 0.01:
            A21 = 0
            phi_1phi1 = 0
        if A11 < 0.01:
            A11 = 0
            phi_2phi1 = 0
        if phi_2phi1 < -0.5:
            phi_2phi1 += 180



    if count > -1:
        f1.write('%s %d %.2f %d %.2f\n'%(station, phi_2phi, A1, phi_1phi, A2))
        if is_WLS:
            plotR(omega, s, xmax, ymax,dt, phi[count], deltat[count], m[0], A1, A2, phi_2phi, phi_1phi, rfr_baz, baz, time_axis, val=val, enf=50, xlim=[3, 7])
        else:
            plotR(dt, phi[count], deltat[count], m[0], A1, A2, phi_2phi, phi_1phi, rfr_baz, baz, time_axis, val=val, enf=50, xlim=[3, 7])
    else:
        time_axis = np.arange(-10, 20, 0.01)
        if is_WLS:
            plotR(floc, sloc, omega, s, xmax, ymax,dt, 0, 0, m[0], A1, A2, phi_2phi, phi_1phi, rfr_baz, baz, time_axis, val=val, enf=50, xlim=[-1, 6.5])
        else:
            plotR(dt, 0, 0, m[0], A1, A2, phi_2phi, phi_1phi, rfr_baz, baz, time_axis, val=val, enf=50, xlim=[-1, 6.5])
    if is_showFigure:
        plt.show()
    return A1_, phi_1phi_, A2_, phi_2phi_,A11, phi_1phi1, A21, phi_2phi1


def cal_omega_mat(rfr_baz, dt):
    evnum = dt.shape[0]
    window_len = np.zeros(evnum)
    dt_height = np.zeros(evnum)
    s_all = np.zeros(evnum)
    omega = np.zeros(evnum)
    ymaxs = np.zeros(evnum)
    xmaxs = np.zeros(evnum)
    ss = np.zeros(evnum)
    floc = np.zeros(evnum)
    sloc = np.zeros(evnum)
    for ev in range(evnum):
        dt_pts = int(1000 + dt[ev] * 100)
        try:
            Rsac = rfr_baz[ev]
        except:
            Rsac = rfr_baz.copy()
        dt_height[ev] = Rsac[int(dt_pts)]
        data_length = Rsac.shape[0]
        times = (Rsac[:(data_length - 1)] - 0.000000001) * Rsac[1:data_length]


        #------Pick the nearest zero points on both sides of the selected PmS signal-----#
        # all_zero_loc = np.where(times < 0)[0]
        # first_zeros = np.argwhere(abs(all_zero_loc - dt_pts) == abs(all_zero_loc - dt_pts).min())[0]
        # if len(first_zeros) == 1:
        #     first_zero_loc = first_zeros[0]
        #     all_tmp = all_zero_loc.copy()
        #     all_tmp[first_zero_loc] = 1000
        #     sec_zero_loc = np.argmin(abs(all_tmp - dt_pts))
        #     if first_zero_loc > sec_zero_loc:
        #         tmp = first_zero_loc
        #         first_zero_loc = sec_zero_loc
        #         sec_zero_loc = tmp
        # else:
        #     first_zero = first_zeros.min()
        #     sec_zero = first_zeros.max()
        # first_zero = all_zero_loc[first_zero_loc]
        # sec_zero = all_zero_loc[sec_zero_loc]
        Rsac_more = Rsac.copy()
        # Rsac_more=np.append(Rsac_more, -10000)
        # Rsac_more=np.insert(Rsac_more, 0, -10000)

        # for pt in range(dt_pts, start, -1):
        #     if (Rsac_more[pt] > Rsac_more[pt-1] and Rsac_more[pt] < Rsac_more[pt+1]) or Rsac_more[pt] * Rsac_more[pt-1] <0:
        #         first_zero = pt
        #         continue
        # for pt in range(dt_pts+10, end):
        #     if (Rsac_more[pt] < Rsac_more[pt-1] and Rsac_more[pt] > Rsac_more[pt+1]) or Rsac_more[pt] * Rsac_more[pt+1] <0:
        #         sec_zero = pt
        #         continue
        first_zero = 0
        sec_zero = 0
        for pt in range(dt_pts-10, start, -1):
            if (Rsac_more[pt] - 0.0001)<=Rsac_more[pt-1] or Rsac_more[pt]*Rsac_more[pt-1]<=0:
                first_zero = pt
                break
               
        for pt in range(dt_pts+10, end):
            if (Rsac_more[pt] - 0.0001)<=Rsac_more[pt+1] or Rsac_more[pt] * Rsac_more[pt+1]<=0:
                sec_zero = pt
                break
        if first_zero == 0:
            first_zero = start
        if sec_zero == 0:
            sec_zero = end
        # first_zero = dt_pts - 30
        # sec_zero = dt_pts + 30

        #------Select a time window for the picked dt of each event------#
        # Rsac_new = -Rsac + 1
        # minus_peak, _ = find_peaks(Rsac_new)
        # pnum = len(minus_peak)
        # window_peak = []
        # for i in range(pnum):
        #     if minus_peak[i] - first_zero > 0 and minus_peak[i] - sec_zero < 0:
        #         window_peak.append(minus_peak[i])
        # w_num = len(window_peak)
        # if w_num > 0:
        #     window_peak = np.array(window_peak)
        #     loc_dt_min = np.where(window_peak - dt_pts < 0)[0]
        #     loc_dt_max = np.where(window_peak - dt_pts > 0)[0]
        #     len_min = len(loc_dt_min)
        #     len_max = len(loc_dt_max)
        #     if len_min > 0 and len_max == 0:
        #         loc_tmp = np.argmin(abs(window_peak - dt_pts))
        #         first_zero = window_peak[loc_tmp]
        #     elif len_min == 0 and len_max > 0:
        #         loc_tmp = np.argmin(abs(window_peak - dt_pts))
        #         sec_zero = window_peak[loc_tmp]
        #     elif len_min * len_max > 0:
        #         loc_tmp1 = loc_dt_min[-1]
        #         first_zero = window_peak[loc_tmp1]
        #         loc_tmp2 = loc_dt_max[0]
        #         sec_zero = window_peak[loc_tmp2]

        #------Calculate the length of the selected time window------#
        # window_len[ev] = sec_zero - first_zero
        # if window_len[ev]==0:
        #     window_len[ev] = 200

    #------Make the omega coefficients based on the lengths of the time windows------#
    # hw = dt_height / window_len 
    # hw_middle = sqrt(hw.max() **2 + hw.min()**2) * 0.5 * window_thresh
    # omega = np.zeros(evnum)
    # for i in range(evnum):
    #     if hw[i] > hw_middle:
    #         omega[i] = 1
    #     else:
    #         omega[i] = (hw[i] / hw_middle) 
    # k = np.zeros(evnum)
    # for i in range(evnum):
    #     k1 = rfr_baz[i,int(100 * dt[i] + 1000)] - rfr_baz[i,int(100 * dt[i] + 999)]
    #     k2 = rfr_baz[i,int(100 * dt[i] + 1001)] - rfr_baz[i,int(100 * dt[i] + 1000)]
    #     k[i] = (k1 - k2) / 2
    # k_norm = k / k.max()
    # omega = np.zeros(evnum)
    # for i in range(evnum):
    #     if k_norm[i] > 0.3:
    #         omega[i] = 1
    #     else:
    #         omega[i] = k_norm[i] / 0.3

        #Calculate the omega based on the Guass curve-fit
        # print(first_zero)
        Xoriginal = np.arange(first_zero *0.01 - 10, sec_zero *0.01 - 10 + 0.01, 0.01)
        # Xoriginal = np.arange(dt[ev] - 1, dt[ev]+1, 0.01)
        x = Xoriginal
        Yoriginal = x.copy()
        yy= x.copy()
        for i in range(len(x)):
            yy[i]=Rsac[int(1000 + x[i] * 100)]
            Yoriginal[i] = np.log(Rsac[int(1000 + x[i] * 100)])
        zmatrix = matrix(matrix(Yoriginal).T)
        xmatrix_T = matrix(np.reshape(np.concatenate((np.ones((len(Yoriginal))), x, x*x)), (3, len(Yoriginal))))
        xmatrix = matrix(xmatrix_T.T)
        bmatrix = (xmatrix_T * xmatrix).I * xmatrix_T * zmatrix
        b0, b1, b2 = float(bmatrix[0][0]), float(bmatrix[1][0]), float(bmatrix[2][0])
        # print(first_zero *0.01 - 10)
        # print(sec_zero *0.01 - 10)
        s = -1 / b2
        xmax = b1 * s / 2
        ymax = exp(b0 + xmax**2 / s)
        y_fit = Yoriginal.copy()
        for i in range(len(x)):
            y_fit[i] = ymax * exp(-(x[i] - xmax)**2 / s)
        xcc = xcorr(yy, y_fit)
        #ppot, pcov = curve_fit(guassian, np.transpose(x), np.transpose(Yoriginal), p0=[ymax, xmax, s])
        #ycov = guassian(x, *ppot)
        # s_all[ev] = abs(2 * ymax / s)
        # print(np.argmax(xcc))
        s_all[ev] = ( np.argmax(xcc) - int(len(x) / 2) ) / 100
        ss[ev] = s 
        xmaxs[ev] = xmax
        ymaxs[ev] = ymax
        floc[ev] = first_zero
        sloc[ev] = sec_zero
        # plt.plot(xcc)
        # plt.plot(y_fit)
        # plt.plot(Rsac[first_zero:sec_zero])
        # plt.plot(yy)
        # plt.show()
    s_norm = 1 / (np.sqrt(abs(s_all)) + 1)
    # print(s_all)
    s_norm = s_norm / s_norm.max()

    k = np.zeros(evnum)
    for i in range(evnum):
        k1 = rfr_baz[i,int(100 * dt[i] + 1000)] - rfr_baz[i,int(100 * dt[i] + 999)]
        k2 = rfr_baz[i,int(100 * dt[i] + 1001)] - rfr_baz[i,int(100 * dt[i] + 1000)]
        k[i] = (k1 - k2) / 2
    k_norm = abs(k / k.max())
    omega = np.zeros(evnum)
    for i in range(evnum):
        if k_norm[i] > 0.7:
            omega[i] = 1
        else:
            omega[i] = abs(k_norm[i])
        # omega[i] = 1
        # s_norm[i] = 1
    for i in range(evnum):
        if s_norm[i]>0.7:
            s_norm[i] = 1

    return (s_norm + omega)/2, ss, xmaxs, ymaxs, floc, sloc

def gaussian(x, ymax, xmax, s):
    return ymax * np.exp(- np.power(x - xmax, 2.) / s)


#------The parameter PmS_loc indicates which dt to pick------#
def pickdt(start, end, rfr_baz, PmS_loc):
    if rfr_baz.shape[0] > 1100:
        Rsac = rfr_baz[start:end]
        evnum = 1
    else:
        Rsac = rfr_baz[:,start:end]
        evnum = Rsac.shape[0]
    peak_loc = np.zeros(evnum)
    for i in range(evnum):
        if evnum > 1:
            peak, _ = find_peaks(Rsac[i])
        else:
            try:
                peak, _ = find_peaks(Rsac.T)
            except:
                peak, _ = find_peaks(Rsac[0])
        peak = np.array(peak)
        amp = np.zeros(peak.shape[0])
        for j in range(peak.shape[0]):
            if evnum > 1:
                amp[j] = Rsac[i, peak[j]]
            else:
                try:
                    amp[j] = Rsac[peak[j]]
                except:
                    amp[j] = Rsac[0,peak[j]]
        try:
            if PmS_loc == 0 and is_refPmS == 0:
                loc = np.argmax(amp)
            elif is_refPmS:
                loc = np.argmin(abs(peak + (start - 1000) / 100 - refPms))
            else:
                amp_tmp = amp.copy()
                while PmS_loc > 0:
                    amp_tmp[np.argmax(amp_tmp)] = 0
                    loc = np.argmax(amp_tmp)
                    PmS_loc -= 1
        except:
            continue
        peak_loc[i] = peak[loc]
    dt = 0.01 * (int(start - 1000) + peak_loc)
    return dt


if __name__ == '__main__':
    start = 1300
    end = 1600
    pt = 3000
    is_boot = 0
    #Whether use the synthetic data or not (the synthetic data is not sac) and whether add Gaussian noise
    is_syn = 1
    is_noise = 0
    #------0/1: Whether show the RF figures and cosine lines or not------#
    is_showFigure = 1
    #------0/1: Whether pick the dts based on the reference Pms arrival time or not------#
    is_refPmS = 0 
    refPms = 5.5
    #------0/1: Whether Use WLS or OLS to solve the cosine line------#
    is_WLS = 1
    #------The threshold for calculating the weighting matrix, should be 0~1------#
    window_thresh = 0.5
    #------The threshold for the iteration of the calcalation of the cosine line------#
    thresh = 0.8
    #------The interval for stacking the RF-----#
    val = 10
    wrong_stalst = []
    lisan_stalst = []
    if is_syn == 0:
        sta_lst = np.loadtxt('RFsta.lst', dtype = str)
        sta_lst = ['14839']
        f1 = open('./lsqr_wls/anis_wls.dat', 'w')
        file = '/Users/jicong/Desktop/RFresult/shanxi/anistropy/anis.dat'
        f2 = open (file)
        f1_lines = f2.readlines()
        sta = []
        phi = []
        deltat = []
        for line in f1_lines:
            line = line.strip()
            line = line.split(' ')
            sta.append(line[0])
            phi.append(float(line[1]))
            deltat.append(float(line[2]))
        phi = np.array(phi)
        deltat = np.array(deltat)
        count = 0
        data_path = '/Users/jicong/Desktop/RFresult'
        for station in sta_lst:
            #Rsac1, baz, _, _= read_data(data_path, station)
            #Rsac = Rsac1[0:int(Rsac1.shape[0] / 2),start:end]
            cal_ani(count)
            count += 1
        f1.close()
    else:
        # sum = np.zeros([8])
        # times = 50
        # result = np.zeros([8, times])
        count = -1
        station = 'Synthetic'
        cal_ani(count)
        # for i in range(times):
        #     A1, phi1, A2, phi2, A11, phi11, A21, phi21= cal_ani(count)
        #     result[0, i] = A1
        #     result[1, i] = phi1
        #     result[2, i] = A2
        #     result[3, i] = phi2
        #     result[4, i] = A21
        #     result[5, i] = phi21
        #     result[6, i] = A11
        #     result[7, i] = phi11
        #     sum[0]+=A1
        #     sum[1]+=phi1
        #     sum[2]+=A2
        #     sum[3]+=phi2            
        #     sum[4]+=A11
        #     sum[5]+=phi11
        #     sum[6]+=A21
        #     sum[7]+=phi21
        #     print(i+1)
        # sum /= times
        # std = np.std(result, axis=1)
        # print('WLS: {:.2f}+{:.2f} {:.2f}+{:.2f} {:.2f}+{:.2f} {:.2f}+{:.2f}'.format(sum[0]/2,std[0]/1.5,sum[1],std[1]/1.2,sum[2],std[2]/2,sum[3],std[3]/1.5))
        # print('OLS: {:.2f}+{:.2f} {:.2f}+{:.2f} {:.2f}+{:.2f} {:.2f}+{:.2f}'.format(sum[6],std[4],sum[7],std[5],sum[4],std[6],sum[5],std[7]))
        # print(A2.mean(), A2.std(), phi2.mean(), phi2.std(), A1.mean(), A1.std(), phi1.mean(), phi1.std())