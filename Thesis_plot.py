#%%
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from Data_sim import DataSimNqubit as DSim
from Linear_inversion import LinearInversion as LI
from QST_nqubit import QStateTomo as QST
from Bayesian_tomo import BayInfTomo as BME
from ExternalFunctions import nice_string_output, add_text_to_ax, Chi2Regression, BinnedLH, UnbinnedLH
from scipy import stats
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator, StrMethodFormatter, NullFormatter
from tqdm import tqdm
from matplotlib import ticker, cm

#%%s
datasim = DSim(1, 100, prob_amp=[1,0],verbose=True, tomo_type='All')
rho_true = datasim.get_density()
expect_dict, expect_values, counts, povm = datasim.measure_qubit()
expect_values_MLE = expect_values[1:]

lin_inv = LI(1, expect_values)
qst = QST(expect_values_MLE,1)
bay_inf_tomo = BME(counts, povm, 1, 50000)

rho_LI = lin_inv.get_lin_inv_rho()
rho_QST = qst.getRho()
rho_BME,theta_distrib_burn, theta_distrib_samp = bay_inf_tomo.getRho()

rho_true_xz = Qobj(np.array(rho_true).real)
rho_LI_xz = Qobj(np.array(rho_LI).real)
rho_QST_xz = Qobj(np.array(rho_QST).real)
rho_BME_xz = Qobj(np.array(rho_BME).real)
rhos = [rho_true_xz, rho_LI_xz, rho_QST_xz, rho_BME_xz]
#%%
import inspect
def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]
def get_bloch_coord(rho):
    sig_x = (rho[1,0] + rho[0,1]).real
    sig_z = (rho[0,0]-rho[1,1]).real
    return sig_x, sig_z
def likelihood(x, z, shots, counts,povm):
    N_fac = np.sum([np.log(n) for n in np.arange(1,shots+1)])
    n_fac = np.sum([np.sum([np.log(n) for n in np.arange(1,i+1)]) for i in counts])
    norm = N_fac-n_fac
    rho_like = 0.5*(qeye(2)+x*sigmax()+z*sigmaz())
    like = np.prod([(po*rho_like).tr()**counts[j] for j,po in enumerate(povm)])
    return np.exp(norm)*like
vfunc = np.vectorize(likelihood,excluded=[2,3,4])
x = np.linspace(-1.5,1.5,100)
z = np.linspace(-1.5,1.5,100)
X, Z = np.meshgrid(x,z)
Y = vfunc(X,Z, 100, counts, povm)
#%%
fig, ax = plt.subplots(figsize=(19,16))
levels = MaxNLocator(nbins=50).tick_values(Y.min(), Y.max())
cmap = 'inferno'
cf = ax.contourf(X ,
                Z , Y, levels=levels,
                cmap=cmap#,locator=ticker.LogLocator()
                )
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('4')  
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_xlabel(r'$\langle\sigma_{x}\rangle$', fontsize=16, loc='right')
ax.set_ylabel(r'$\langle\sigma_{z}\rangle$', fontsize=16, loc='top' )
ax.set_title('Comparison of different forms of Quantum Tomography ')

# Move left y-axis and bottim x-axis to centre, passing through (0,0)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

for i in rhos:
    sig_x, sig_z = get_bloch_coord(i)
    print(sig_x,sig_z)
    var_name = retrieve_name(i)[1]
    ax.scatter(sig_x,sig_z,label=var_name)

#circle = plt.Circle((0,0), 1, color='white',alpha=0.3, edgecolor='white')
#ax.add_patch(circle)
fig.colorbar(cf, ax=ax, label='Trace distance')
fig.legend()
fig.tight_layout()
#%%
print(rho_LI)
print(rho_QST)
print(rho_BME)
print(f"Linear Inversion fidelity: {tracedist(rho_true, rho_LI)}.")
print(f"Quantum State Tomography fidelity: {tracedist(rho_true, rho_QST)}.")
print(f"Bayesian Inference Tomography fidelity: {tracedist(rho_true, rho_BME)}.")
#%%
rho_like = [[0.5*(qeye(2)+x[i]*sigmax()+z[j]*sigmaz()) for i in range(len(x))] for j in range(len(z))]
#%%

n_fac = sum([sum([np.log(n) for n in np.arange(1,i+1)]) for i in counts])
#%%
x = np.arange(1,counts[0]+1,dtype = int)
print(type(x[0]))
#%%
# %%
print(((povm_list[0]*basis(2,0).proj()).tr())**5)
#%%
x = likelihood(1.5,1.5,10,counts,povm)
print(x)
#%%
from math import factorial as fac
norma = fac(10)/(fac(6)*fac(4)*fac(6)*fac(4)*fac(10)*fac(0))
print(norma)
rho = 0.5*(qeye(2)+0*sigmax()+0*sigmaz())
print(rho)
like = ((povm[0]*rho).tr()**6)*((povm[1]*rho).tr()**4)*((povm[2]*rho).tr()**6)*((povm[3]*rho).tr()**4)*((povm[4]*rho).tr()**10)*((povm[5]*rho).tr()**0)
print(like)
print(norma*like)