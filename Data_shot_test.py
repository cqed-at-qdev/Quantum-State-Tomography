#%%
# Packages
import collections
from sys import set_coroutine_origin_tracking_depth

from numpy.lib.function_base import _quantile_dispatcher
from scipy.optimize.optimize import _prepare_scalar_function
import qutip
from qutip import *
from itertools import zip_longest
import numpy as np
import scipy
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from Data_generation import Datagen
from QST_1qubit import QStateTomo
import Functions as f
import ExternalFunctions as ef
import settings as s
from iminuit import Minuit
s.init()
#%%
a = 1/np.sqrt(3)
b = np.exp(1j*np.pi/8)
s.rho_true = f.get_rho_true(a,b)
Paulis = [sigmax(), sigmay(), sigmaz()]
shot_list = [10,50,100,500,1000,5000,10000,50000,100000,500000]
state_list = [Datagen(a,b,i) for i in shot_list]
data_list = [i.getdatafile() for i in state_list]
Tomo_list = [QStateTomo(data_list[i],Paulis) for i in range(len(data_list))]
fig, axs = plt.subplots(5,2, figsize=(8,15), sharex=False, sharey=False)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('MLE step')
plt.ylabel('Trace Distance')
plt.title('Plots of trace distance vs MLE step for different # shots', pad=30)
fig.tight_layout()
plot_list = [[0,0],[0,1],[1,0],[1,1],[2,0],[2,1],[3,0],[3,1],[4,0],[4,1]]
plot_list_1 = [0,0,1,1,2,2,3,3,4,4]
plot_list_2 = [0,1,0,1,0,1,0,1,0,1]
for i in range(len(Tomo_list)):
    rho_i = Tomo_list[i].getRho()
    axs[plot_list_1[i],plot_list_2[i]].set_yscale('log')
    axs[plot_list_1[i],plot_list_2[i]].set(title=f'{shot_list[i]}shots')
    d = {'sigx =': data_list[i][0],
     'sigy =': data_list[i][1],
     'sigz =': data_list[i][2],
     'final dist =': s.tracedist_list[-1]
    }
    text = ef.nice_string_output(d, extra_spacing=2, decimals=3)
    ef.add_text_to_ax(0.3, 0.95, text, axs[plot_list_1[i],plot_list_2[i]], fontsize=10)
    axs[plot_list_1[i],plot_list_2[i]].plot(s.tracedist_list, marker='o')
    axs[plot_list_1[i],plot_list_2[i]].set_xlim(0,20)
    axs[plot_list_1[i],plot_list_2[i]].set_ylim(min(s.tracedist_list),1)
    s.tracedist_list=[]
#%%
a = 1/np.sqrt(3)
b = np.exp(1j*np.pi/8)
s.rho_true = f.get_rho_true(a,b)
Paulis = [sigmax(), sigmay(), sigmaz()]
shot_list = [10,50,100,500,1000,5000,10000,50000,100000,500000]
state_list = [Datagen(a,b,i) for i in shot_list]
fig, axs = plt.subplots(5,2, figsize=(10,20), sharex=False, sharey=False)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('MLE step')
plt.ylabel('Trace Distance')
plt.title('Plots of trace distance vs MLE step for different # shots with 100 trials', pad=30)
fig.tight_layout()
plot_list = [[0,0],[0,1],[1,0],[1,1],[2,0],[2,1],[3,0],[3,1],[4,0],[4,1]]
plot_list_1 = [0,0,1,1,2,2,3,3,4,4]
plot_list_2 = [0,1,0,1,0,1,0,1,0,1]
trials = 10
for i in range(len(shot_list)):
    trace_dist_trials = np.empty((20,trials))
    trace_dist_trials[:] = np.nan
    data_master = np.zeros((3, trials))
    for x in range(trials):
        data_list = state_list[i].getdatafile()
        data_master[:3,x] = data_list
        Tomo_list = QStateTomo(data_list,Paulis)
        rho_i = Tomo_list.getRho()
        s.tracedist_list = np.trim_zeros(np.array(s.tracedist_list[:20]))
        trace_dist_trials[:len(s.tracedist_list),x] = s.tracedist_list
        s.tracedist_list = []
    mean_trace_dist_trials = np.nanmedian(trace_dist_trials, axis=1)
    print(trace_dist_trials) 
    trace_dist_trials = np.transpose(trace_dist_trials)
    axs[plot_list_1[i],plot_list_2[i]].set_yscale('log')
    axs[plot_list_1[i],plot_list_2[i]].set(title=f'{shot_list[i]}shots')

    d = {'sigx av =': np.mean(data_master[0]),
     'sigy av =': np.mean(data_master[1]),
     'sigz av =': np.mean(data_master[2]),
     'final dist =': mean_trace_dist_trials[-1]
    }
    text = ef.nice_string_output(d, extra_spacing=2, decimals=3)
    ef.add_text_to_ax(0.3, 0.95, text, axs[plot_list_1[i],plot_list_2[i]], fontsize=10)
    
    axs[plot_list_1[i],plot_list_2[i]].plot(mean_trace_dist_trials, marker='o')
    axs[plot_list_1[i],plot_list_2[i]].boxplot(trace_dist_trials, showfliers=False, positions=[int(x) for x in np.linspace(0,19,20)])
    axs[plot_list_1[i],plot_list_2[i]].set_ylim(1e-4,1)
    s.tracedist_list=[]
#%%
a = 1/np.sqrt(3)
b = np.exp(1j*np.pi/8)
s.rho_true = f.get_rho_true(a,b)
Paulis = [sigmax(), sigmay(), sigmaz()]
s.tracedist_list = []
final_trace_list = []
shot_list = [int(x) for x in range(100,50100,100)]
state_list = [Datagen(a,b,i) for i in shot_list]
trials_2 = 10
for i in range(len(shot_list)):
    trace_mean_list = []
    for x in range(trials_2):
        data_list = state_list[i].getdatafile()
        Tomo_list = QStateTomo(data_list,Paulis)
        rho_i = Tomo_list.getRho()
        trace_mean_list.append(s.tracedist_list[-1])
        s.tracedist_list = [] 
    trace_mean = np.mean(trace_mean_list)
    final_trace_list.append(trace_mean)
    s.tracedist_list = []
#%%
fig, ax = plt.subplots(figsize=(16,10))
ax.set_xlabel('# Shots')
ax.set_ylabel('Final trace distance')
ax.set_title('Final trace distance as function of number of shots', pad=30)
#ax.set_yscale('log')
ax.set_xscale('log')
#ax.set_yticklabels(["$%.1e$" %y for y in yticks])
ax.plot(shot_list,final_trace_list, marker='o')
ax.plot(x, fit_func(x, *minuit.values[:]), '-r',label='Power fit')     # Note how we can "smartly" input the fit values!
ax.grid()
d = {'a':   [a_fit, sa],
     'b':   [b_fit, sb],
     'Chi2':     Chi2_fit,
     'ndf':      Ndof_fit,
     'Prob':     Prob_fit,
    }
text = ef.nice_string_output(d, extra_spacing=2, decimals=3)
ef.add_text_to_ax(0.8,0.99, text, ax, fontsize=10)
fig.tight_layout()
#%%
Npoints = len(shot_list)
x = np.arange(50000, step=0.3)

def fit_func(x,a,b):
    return b*x**a

def chi2(a,b):
    y_fit = fit_func(x,a,b)
    chi2 = np.sum(((final_trace_list-y_fit))**2)
    return chi2
popt, pcov = scipy.optimize.curve_fit(fit_func, shot_list, final_trace_list)
residuals = fit_func(shot_list, *popt) - final_trace_list
std = np.std(residuals)
chi2.errordef = 1.0
ef.Chi2Regression.errordef = 1.0
chi2_reg = ef.Chi2Regression(fit_func, shot_list, final_trace_list, sy=std)
minuit = Minuit(chi2_reg, a=-0.001, b=1)
# Perform the actual fit:
minuit.migrad();
    
# Extract the fitting parameters and their errors:
a_fit = minuit.values['a']
b_fit = minuit.values['b']
sa = minuit.errors['a']
sb = minuit.errors['b']
Nvar = 2
Ndof_fit = Npoints - Nvar
Chi2_fit = minuit.fval                         
Prob_fit = stats.chi2.sf(Chi2_fit, Ndof_fit)
fig, ax = plt.subplots(figsize=(12,8))
ax.set_xlabel('# Shots')
ax.set_ylabel('Final trace distance')
ax.set_title('Final trace distance as function of number of shots', pad=30)
ax.set_yscale('log')
ax.set_xscale('log')
ax.axis('equal')
ax.set_ylim(0.001,0.1)
ax.set_xlim(100,50000)
#ax.set_yticklabels(["$%.1e$" %y for y in yticks])
ax.plot(shot_list,final_trace_list, marker='o')
ax.plot(x, fit_func(x, *minuit.values[:]), '-r',label='Power fit')     # Note how we can "smartly" input the fit values!
ax.grid()
d = {'a':   [a_fit, sa],
     'b':   [b_fit, sb],
     'Chi2':     Chi2_fit,
     'ndf':      Ndof_fit,
     'Prob':     Prob_fit,
    }
text = ef.nice_string_output(d, extra_spacing=2, decimals=3)
ef.add_text_to_ax(0.8,0.99, text, ax, fontsize=10)
fig.tight_layout()
print(a_fit)
print(Prob_fit)
print(Chi2_fit)
#%%

#%%
fig, ax = plt.subplots()
ax.set_xlabel('# Shots')
ax.set_ylabel('Final trace distance')
ax.set_title('Final trace distance as function of number of shots', pad=30)
ax.set_yscale('log')
#ax.set_xscale('log')
yticks = [0.001,0.002,0.004,0.008,0.016,0.032, 0.064]
ax.set_yticks(yticks)
ax.set_yticklabels(["$%.1e$" %y for y in yticks])
ax.plot(shot_list[:100],final_trace_list[:100], marker='o')
fig.tight_layout()
#%%
fig, ax = plt.subplots()
ax.set_xlabel('# Shots')
ax.set_ylabel('Final trace distance')
ax.set_title('Final trace distance as function of number of shots', pad=30)
ax.set_yscale('log')
#ax.set_xscale('log')
yticks = [0.001,0.002,0.004,0.008,0.016,0.032, 0.064]
ax.set_yticks(yticks)
ax.set_yticklabels(["$%.1e$" %y for y in yticks])
ax.plot(shot_list[:25],final_trace_list[:25], marker='o')
fig.tight_layout()
#%%
s.tracedist_list = []
state_names = ['Z_cardinal_0', 'X_cardinal_+', 'Y_cardinal_i+', 'H_eigenstate_+', 'Magic_state']
a_list = [1, 1/np.sqrt(2), 1/np.sqrt(2), 1+np.sqrt(2), np.cos(np.arccos(1/np.sqrt(3))/2)]
b_list = [0, 1/np.sqrt(2), 1j/np.sqrt(2), 1, np.exp(1j*np.pi/4)*np.sin(np.arccos(1/np.sqrt(3))/2)]
trials=10
shots = 10000
Paulis = [sigmax(), sigmay(), sigmaz()]
def gen_many_state(a, b, state, trials, shots, Paulis):
    trace_mean_list_plot = []
    for i in range(len(state)):
        trace_dist_trials = np.empty((150,trials))
        trace_dist_trials[:] = np.nan
        for x in range(trials):
            qstate = Datagen(a[i],b[i], shots)
            data = qstate.getdatafile()
            tomo = QStateTomo(data, Paulis)
            rho = tomo.getRho()
            trace_dist_trials[:len(s.tracedist_list),x] = s.tracedist_list
            s.tracedist_list = []
        trace_mean_list = np.nanmean(trace_dist_trials, axis=1)
        trace_mean_list = trace_mean_list[~np.isnan(trace_mean_list)]
        trace_mean_list_plot.append(trace_mean_list)
    return trace_mean_list_plot
def plot_many_state(trace_mean_list_plot, state_names):
    fig, ax = plt.subplots(figsize=(16,10))
    ax.set_xscale('log')
    ax.set_yscale('log')
    for col,i in zip(trace_mean_list_plot,state_names):
        ax.plot(col, label=i)
    fig.legend()
    fig.tight_layout()
#%%
x = gen_many_state(a_list,b_list,state_names,trials,shots,Paulis)
#%%
plot = plot_many_state(x, state_names)

#%%
#%%
s.tracedist_list = []
a_list = [1,1+np.sqrt(2)]
b_list = [0,1]
trials = 10
shots = 10000
Paulis = [sigmax(), sigmay(), sigmaz()]
def gen_many_state(a, b, trials, shots, Paulis):
    trace_mean_list_plot = []
    trace_std_list_plot = []
    for i in range(len(a)):
        trace_dist_trials = []
        qstate = Datagen(a[i], b[i], shots)
        data = [qstate.getdatafile() for _ in range(trials)]
        tomo = [QStateTomo(x, Paulis) for x in data]
        for x in range(trials):
            rho = tomo[x].getRho()
            trace_dist_trials.append(s.tracedist_list[-1])
            s.tracedist_list = []
        trace_mean = np.mean(trace_dist_trials)
        trace_std = np.std(trace_dist_trials)
        trace_mean_list_plot.append(trace_mean)
        trace_std_list_plot.append(trace_std)
    return trace_mean_list_plot
def plot_many_state(theta,mean):
    fig, ax = plt.subplots(figsize=(16,10))
    ax.set_title('Final trace distance as function of theta')
    ax.set_ylabel('Final trace distance')
    ax.set_xlabel('Theta')
    ax.text(0.9,0.98, 'Shots={}'.format(shots),
    horizontalalignment='center',verticalalignment='center', transform = ax.transAxes)
    ax.text(0.9,0.96, 'Trials={}'.format(trials),
    horizontalalignment='center',verticalalignment='center', transform = ax.transAxes)
    ax.text(0.9,0.94, 
    r'$\left|\psi\right\rangle=\cos(\theta/2)\left|0\right\rangle+\sin(\theta/2)\left|1\right\rangle$',
    horizontalalignment='center',verticalalignment='center', transform = ax.transAxes)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_xticks([0 , np.pi/8,np.pi/6,np.pi/4])
    ax.set_xticklabels(['0', '$\pi/8$','$\pi/6$','$\pi/4$'])
    ax.plot(theta, mean, marker='o', color='r')
    fig.legend()
    return fig, ax
#%%
mean, std = gen_many_state(a_list,b_list,trials,shots,Paulis)
#%%
x = np.linspace(0,np.pi,1000)
def fit_func(x,a,b,c,d):
    return a*np.sin(b*(x-c))+d
popt, pcov = optimize.curve_fit(fit_func, x[:250], mean[:250], p0=[10,10,10,10])
# Calculating residuals
residuals = fit_func(theta_list, *popt) - mean
#%%
fig, ax = plot_many_state(theta_list,mean,std)
#%%
print(mean)
#%%
x = np.linspace(0,0.7853981765,10)
states = [(np.cos(i/2)*basis(2,0)+np.sin(i/2)*basis(2,1))/(np.cos(i/2)*basis(2,0)+np.sin(i/2)*basis(2,1)).norm() for i in x]
bloch = Bloch()
for i in states:
    bloch.add_states(i)
bloch.show()
#%%
first = np.mean(mean[:90])
second = np.mean(mean[90:180])
third = np.mean(mean[180:270])
fourth = np.mean(mean[270:360])
print(first,second,third,fourth)
#%%
theta_list = [x for x in np.linspace(0,np.pi/4,4)]
a_list = [np.cos(x/2) for x in theta_list]
b_list = [np.sin(x/2) for x in theta_list]
theta = [0, np.pi/4]
a = [1,np.cos(theta[1]/2)]
b =[0, np.sin(theta[1]/2)]
trials = 100
shots = 10000
Paulis = [sigmax(), sigmay(), sigmaz()]
#%%
mean = [gen_many_state(a_list, b_list, trials, shots, Paulis) for _ in range(3)]
#%%
for x in range(3):
    plt.plot(theta_list,mean[x])
#%%
fig, ax = [plot_many_state(theta,y) for y in mean]
#%%
np.linspace(0,np.pi/4,4)
#%%
mean