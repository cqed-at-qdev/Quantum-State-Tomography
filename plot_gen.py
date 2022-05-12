#%%
# Packages
import string
from numpy.core.defchararray import center
from sympy.core.numbers import Zero
import qutip
from qutip import *
import numpy as np
import scipy
from sympy import symbols
from sympy.physics.quantum import Bra, Ket
from scipy import stats
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
def trace_dist_trials(a,b,trials):
    s.rho_true = f.get_rho_true(a,b)
    Paulis = [sigmax(), sigmay(), sigmaz()]
    s.tracedist_list = []
    final_trace_list = []
    shot_list = [int(x) for x in range(100,50100,100)]
    state_list = [Datagen(a,b,i) for i in shot_list]
    for i in range(len(shot_list)):
        trace_mean_list = []
        for x in range(trials):
            data_list = state_list[i].getdatafile()
            Tomo_list = QStateTomo(data_list,Paulis)
            rho_i = Tomo_list.getRho()
            trace_mean_list.append(s.tracedist_list[-1])
            s.tracedist_list = [] 
        trace_mean = np.mean(trace_mean_list)
        final_trace_list.append(trace_mean)
        s.tracedist_list = []
    return final_trace_list, shot_list
#%%
def fit_func(x,a,b):
    return b*x**a

def getplot(final_trace_list, shot_list, state, Beta):
    Npoints = len(final_trace_list)
    x = np.arange(50000, step=0.3)
    popt, pcov = scipy.optimize.curve_fit(fit_func, shot_list, final_trace_list)
    residuals = fit_func(shot_list, *popt) - final_trace_list
    std = np.std(residuals)
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
    ## Regular plot
    fig1, ax1 = plt.subplots(figsize=(12,8))
    ax1.set_xlabel('# Shots')
    ax1.set_ylabel('Final trace distance')
    ax1.set_title('Final trace distance as function of number of shots', pad=30)
    ax1.set_ylim(0.001,0.1)
    ax1.set_xlim(100,50000)
    ax1.text(0.5,0.95,state, horizontalalignment='center',verticalalignment='center', transform = ax1.transAxes)
    ax1.text(0.5,0.9,Beta, horizontalalignment='center',verticalalignment='center', transform = ax1.transAxes)
    ax1.plot(shot_list,final_trace_list, marker='o')
    ax1.plot(x, fit_func(x, *minuit.values[:]), '-r',label='Power fit')     # Note how we can "smartly" input the fit values!
    ax1.grid()
    d = {'a':   [a_fit, sa],
        'b':   [b_fit, sb],
        'Chi2':     Chi2_fit,
        'ndf':      Ndof_fit,
        'Prob':     Prob_fit,
        }
    text = ef.nice_string_output(d, extra_spacing=2, decimals=3)
    ef.add_text_to_ax(0.8,0.99, text, ax1, fontsize=10)
    fig1.tight_layout()
    fig1.savefig(r"C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Plots\Magic_nonlog.png")    
    ## Logy plot 
    fig2, ax2 = plt.subplots(figsize=(12,8))
    ax2.set_xlabel('# Shots')
    ax2.set_ylabel('Final trace distance')
    ax2.set_title('Final trace distance as function of number of shots', pad=30)
    ax2.set_yscale('log')
    ax2.set_ylim(0.001,0.1)
    ax2.set_xlim(100,50000)
    ax2.text(0.5,0.95,state, horizontalalignment='center',verticalalignment='center', transform = ax2.transAxes)
    ax2.text(0.5,0.9,Beta, horizontalalignment='center',verticalalignment='center', transform = ax2.transAxes)
    ax2.plot(shot_list,final_trace_list, marker='o')
    ax2.plot(x, fit_func(x, *minuit.values[:]), '-r',label='Power fit')     # Note how we can "smartly" input the fit values!
    ax2.grid()
    d = {'a':   [a_fit, sa],
        'b':   [b_fit, sb],
        'Chi2':     Chi2_fit,
        'ndf':      Ndof_fit,
        'Prob':     Prob_fit,
        }
    text = ef.nice_string_output(d, extra_spacing=2, decimals=3)
    ef.add_text_to_ax(0.8,0.99, text, ax2, fontsize=10)
    fig2.tight_layout()
    fig2.savefig(r"C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Plots\Magic_logy.png")
    ## LogLog plot
    fig3, ax3 = plt.subplots(figsize=(12,8))
    ax3.set_xlabel('# Shots')
    ax3.set_ylabel('Final trace distance')
    ax3.set_title('Final trace distance as function of number of shots', pad=30)
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.axis('equal')
    ax3.set_ylim(0.001,0.1)
    ax3.set_xlim(100,50000)
    ax3.text(0.5,0.95,state, horizontalalignment='center',verticalalignment='center', transform = ax3.transAxes)
    ax3.text(0.5,0.9,Beta, horizontalalignment='center',verticalalignment='center', transform = ax3.transAxes)
    ax3.plot(shot_list,final_trace_list, marker='o')
    ax3.plot(x, fit_func(x, *minuit.values[:]), '-r',label='Power fit')     # Note how we can "smartly" input the fit values!
    ax3.grid()
    d = {'a':   [a_fit, sa],
        'b':   [b_fit, sb],
        'Chi2':     Chi2_fit,
        'ndf':      Ndof_fit,
        'Prob':     Prob_fit,
        }
    text = ef.nice_string_output(d, extra_spacing=2, decimals=3)
    ef.add_text_to_ax(0.8,0.99, text, ax3, fontsize=10)
    fig3.tight_layout()
    fig3.savefig(r"C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Plots\Magic_logxy.png")
#%%
a=np.cos(np.arccos(1/np.sqrt(3))/2)
b=np.exp(1j*np.pi/4)*np.sin(np.arccos(1/np.sqrt(3))/2)
final_trace_list, shot_list = trace_dist_trials(a,b,10)
#%%
a=np.cos(np.arccos(1/np.sqrt(3))/2)
b=np.exp(1j*np.pi/4)*np.sin(np.arccos(1/np.sqrt(3))/2)
state = r'$\left|\psi\right\rangle=\cos{\frac{\beta}{2}}\cdot\left|0\right\rangle+\exp{\frac{i\pi}{4}}\sin{\frac{\beta}{2}}\cdot\left|1\right\rangle$'
Beta = r'$\beta=\arccos(\frac{1}{\sqrt{3}})$'
getplot(final_trace_list,shot_list, state, Beta)
#%%
a=1/np.sqrt(4-2*np.sqrt(2))
b=1/np.sqrt(2*np.sqrt(2))
state = r'$\left|\psi\right\rangle=\cos{\frac{\beta}{2}}\cdot\left|0\right\rangle+\exp{\frac{i\pi}{4}}\sin{\frac{\beta}{2}}\cdot\left|1\right\rangle$'
Beta = r'$\beta=\arccos(\frac{1}{\sqrt{3}})$'
fig, ax = plt.subplots()
ax.text(0.5,0.5,state)
ax.text(0.5,0.40,Beta)
#%%