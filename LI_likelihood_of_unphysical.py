#%%
from ast import literal_eval
import numpy as np
from qutip import *
from Linear_inversion import LinearInversion
from Data_sim import DataSimNqubit
import csv
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from iminuit import Minuit
from ExternalFunctions import nice_string_output, add_text_to_ax, Chi2Regression, BinnedLH, UnbinnedLH
from scipy import stats
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator, StrMethodFormatter, NullFormatter
from tqdm import tqdm
#%%
n_states = 280
n_qubits = [3]
n_eval = 0
df = pd.read_pickle("Inverted_hadamards.pkl")
df1 = pd.read_pickle("datasim_keylist.pkl")
for q in n_qubits:
    shots = [3**x for x in np.arange(9,14)]
    expect_key = list(filter(None, df1.values.tolist()[q-1]))
    purity_list = list(np.linspace(1/(2**q),1,25))
    for n in shots:
        for x in purity_list:
            counts = 0
            trace_list = []
            for i in range(n_states):
                sim = DataSimNqubit(q, n, purity=x, explicit=False, df=df, df1=df1, expect_key=expect_key, total_shot=True)
                data = sim.measure_qubit()[-1]
                rho_true = sim.get_density()
                linear_inv_tomo = LinearInversion(q, data, explicit=False, expect_key=expect_key)
                rho_lin = linear_inv_tomo.get_lin_inv_rho()
                trace = tracedist(rho_true,rho_lin)
                trace_list.append(trace)
                n_eval += 1
                print(f"{n_eval/(280*25*12+280*25*11)*100}%")
            trace_avg = np.mean(trace_list)
            f = open(fr'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\LI_data\Fidelity_3D_{q}_high_shot.csv', 'a', newline='')
            writer = csv.writer(f)
            writer.writerow([x, n, trace_avg])
            f.close()
    fig, ax = plt.subplots(figsize=(16,10))
    data = np.genfromtxt(fr'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\LI_data\Fidelity_3D_{q}_high_shot.csv',delimiter=',')
    x=data[:,0]
    y=data[:,1]
    z=data[:,2]
    x=np.unique(x)
    y=np.unique(y)
    X,Y = np.meshgrid(x,y)
    Z=z.reshape(len(y),len(x))
    levels = MaxNLocator(nbins=50).tick_values(Z.min(), Z.max())
    cmap = 'inferno'
    norm = BoundaryNorm(levels, ncolors=50, clip=True)
    cf = ax.contourf(X ,
                    Y , Z, levels=levels,
                    cmap=cmap)
    ax.set_title(f'2D heatmap of probability of LI giving a unphysical state - {q} Qubits')
    ax.set_xlabel('Purity')
    ax.set_ylabel('Shots')
    ax.set_yscale('log')
    ax.set_xlim(0,1)
    ax.set_ylim(3,max(y))
    fig.colorbar(cf, ax=ax, label='Probability')
    fig.tight_layout()
#%%
# Create regular heat map
from matplotlib import ticker, cm
fig, ax = plt.subplots(figsize=(16,10))
data = np.genfromtxt(fr'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\LI_data\Fidelity_3D_1_high_shot.csv',delimiter=',')
x=data[:,0]
y=data[:,1]
z=data[:,2]
x=np.unique(x)
y=np.unique(y)
X,Y = np.meshgrid(x,y)
Z=z.reshape(len(y),len(x))
levels = MaxNLocator(nbins=50).tick_values(Z.min(), Z.max())
cmap = 'inferno'
norm = BoundaryNorm(levels, ncolors=50, clip=True)
cf = ax.contourf(X ,
                Y , Z, levels=levels,
                cmap=cmap, locator=ticker.LogLocator())
ax.set_title(f'2D heatmap of LI fidelity - 3 Qubits')
ax.set_xlabel('Purity')
ax.set_ylabel('Shots')
ax.set_yscale('log')
ax.set_xlim(0,1)
ax.set_ylim(3,max(y))
fig.colorbar(cf, ax=ax, label='Trace distance')
fig.tight_layout()
#%%
fig, ax = plt.subplots(figsize=(16,10))
data = np.genfromtxt(r'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\Purity_3D_2_high_res.csv',delimiter=',')
x=data[:,0]
x = 1 - np.sqrt(1-((1-x)/(1-(1/(2**2)))))
y=data[:,1]
z=data[:,2]
x=np.unique(x)
print(x)
y=np.unique(y)
X,Y = np.meshgrid(x,y)
Z=z.reshape(len(y),len(x))
levels = MaxNLocator(nbins=50).tick_values(Z.min(), Z.max())
cmap = 'inferno'
norm = BoundaryNorm(levels, ncolors=50, clip=True)
cf = ax.contourf(X ,
                  Y , Z, levels=levels,
                  cmap=cmap)
ax.set_title('2D heatmap of probability of LI giving a unphysical state - 2 Qubit')
ax.set_xlabel('Epsilon')
ax.set_ylabel('Shots per measurment (3 measurments)')
ax.set_xlim(0,1)
ax.set_yscale('log')
ax.set_ylim(min(y),max(y))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
ax.yaxis.set_minor_formatter(NullFormatter())
#ax.pcolormesh(X,Y,Z)
fig.colorbar(cf, ax=ax, label='Probability')
fig.tight_layout()
#%%
shots = [3**x for x in np.arange(9,14)]
print(shots)
f = [x/(3**3) for x in shots]
print(f)
#%%
x = tensor(qeye(2),qeye(2),qeye(2)).unit()
print(x)
y = basis(8,0).proj()
print(y)
tracedist(x,y)
#%%
