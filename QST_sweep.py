#%%
import numpy as np
from qutip import *
from QST_nqubit import QStateTomo
from Data_sim import DataSimNqubit
import matplotlib.pyplot as plt
import pandas as pd
import csv
from ExternalFunctions import nice_string_output, add_text_to_ax, Chi2Regression, BinnedLH, UnbinnedLH
from scipy import stats
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator, StrMethodFormatter, NullFormatter
from tqdm import tqdm
def pauli_calculator(nqubits):
        """Takes the row from the datasim_keylist corresponding to n qubits
        and converts it to a set of pauli matrices, which get tensored.
        Ex. if the current key is 'x2x3' it turns into 'tensor(I,sig_x,sig_x).

        Returns:
            Paulis: ([list]): returns list of pauli tensor products
        """
        # Read pickle file into dataframed
        df1 = pd.read_pickle("datasim_keylist.pkl")
        # Remove any None values and grab the correct row
        expect_val = list(filter(None, df1.values[nqubits-1]))
        # Remove duplicates
        expect_key = list(dict.fromkeys(expect_val))
        paulis = []
        # Loop through every key in the list 
        for key in expect_key:
            tup_list = []
            num_array= np.zeros(nqubits)
            # Loop through every substring of the key and append the correct paulis 
            # to the tup_list
            for value in key:
                if 'x' in value:
                    tup_list.append(sigmax())
                if 'y' in value:
                    tup_list.append(sigmay())
                if 'z' in value:
                    tup_list.append(sigmaz())
                # If string is a number, append it to the predefined num_array
                # to the correct spot
                if value != 'x':
                    if value != 'y':
                        if value != 'z':
                            num_array[int(value)-1] = value
                else:
                    pass
            # If tup_list is nqubits long then all is good
            if len(tup_list)==nqubits:
                pauli_set = tensor(tup_list)
                paulis.append(pauli_set)
            # Otherwise insert identities in the correct positions.
            else:
                for i,x in enumerate(num_array):
                    if x==0:
                        tup_list.insert(i,qeye(2))
                    else:
                        pass
                pauli_set = tensor(tup_list)
                paulis.append(pauli_set)
        return paulis, expect_val
#%%
qubit_list = [3]
num_states = 200
tol = [1e-15,1e-10,1e-5]
for n in qubit_list:
    paulis, expect_val = pauli_calculator(n)
    hadamard = pd.read_pickle("Inverted_hadamards.pkl")[0][n-1]
    shots = [3**x for x in np.arange(n,14)]
    purity = list(np.linspace(1/(2**n),1,25))
    for s in shots:
        for p in purity:
            trace_av_list = []
            count_high = 0
            count_med = 0
            count_low = 0
            for x in range(num_states):
                data = DataSimNqubit(n, s, purity=p, explicit=False, tomo_type='MLE', had_exp=hadamard, total_shot=True, expect_key=expect_val)
                rho_true = data.get_density()
                dict_ex, expectation = data.measure_qubit()
                del expectation[0]
                qst = QStateTomo(expectation, n, explicit=False, paulis=paulis)
                rho = qst.getRho()
                trace_av_list.append(tracedist(rho_true, rho))
                eigen = rho.eigenenergies()
                if any(i < tol[0] for i in eigen)==True:
                    count_high += 1
                if any(i < tol[1] for i in eigen)==True:
                    count_med += 1
                if any(i < tol[2] for i in eigen)==True:
                    count_low += 1
                else:
                    pass
            trace_av = np.mean(trace_av_list)
            prob_high = count_high/num_states
            prob_med = count_med/num_states
            prob_low = count_low/num_states

            f = open(fr'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\QST_data\trace_3D_{n}.csv', 'a', newline='')
            writer = csv.writer(f)
            writer.writerow([p,s,trace_av])
            f.close()

            f = open(fr'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\QST_data\eigen_3D_{n}_high_tol.csv', 'a', newline='')
            writer = csv.writer(f)
            writer.writerow([p,s,prob_high])
            f.close()

            f = open(fr'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\QST_data\eigen_3D_{n}_med_tol.csv', 'a', newline='')
            writer = csv.writer(f)
            writer.writerow([p,s,prob_med])
            f.close()

            f = open(fr'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\QST_data\eigen_3D_{n}_low_tol.csv', 'a', newline='')
            writer = csv.writer(f)
            writer.writerow([p,s,prob_low])
            f.close()
#%%
## trace plots
n = 2
fig1, ax1 = plt.subplots(figsize=(16,10))
data = np.genfromtxt(fr'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\QST_data\trace_3D_{n}.csv',delimiter=',')
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
cf = ax1.contourf(X ,
                Y , Z, levels=levels,
                cmap=cmap)
ax1.set_title(f'2D heatmap of fidelity of QST - {n} Qubits')
ax1.set_xlabel('Purity')
ax1.set_ylabel('Total shots')
ax1.set_yscale('log')
ax1.set_xlim(0,1)
ax1.set_ylim(3,max(y))
fig1.colorbar(cf, ax=ax1, label='Trace distance')
fig1.tight_layout()

fig2, ax2 = plt.subplots(figsize=(16,10))
data = np.genfromtxt(fr'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\QST_data\trace_3D_{n}.csv',delimiter=',')
x=data[:,0]
y=data[:,1]/(3**n)
z=data[:,2]
x=np.unique(x)
y=np.unique(y)
X,Y = np.meshgrid(x,y)
Z=z.reshape(len(y),len(x))
levels = MaxNLocator(nbins=50).tick_values(Z.min(), Z.max())
cmap = 'inferno'
norm = BoundaryNorm(levels, ncolors=50, clip=True)
cf = ax2.contourf(X ,
                Y , Z, levels=levels,
                cmap=cmap)
ax2.set_title(f'2D heatmap of fidelity of QST - {n} Qubits')
ax2.set_xlabel('Purity')
ax2.set_ylabel('Shots per basis')
ax2.set_yscale('log')
ax2.set_xlim(0,1)
ax2.set_ylim(1,max(y))
fig2.colorbar(cf, ax=ax2, label='Trace distance')
fig2.tight_layout()

## eigen plots
fig3a, ax3a = plt.subplots(figsize=(16,10))
data = np.genfromtxt(fr'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\QST_data\eigen_3D_{n}_high_tol.csv',delimiter=',')
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
cf = ax3a.contourf(X ,
                Y , Z, levels=levels,
                cmap=cmap)
ax3a.set_title(f'2D heatmap of the probability of QST giving a 0 eigenenergy - {n} Qubits - High zero tolerance: 1e-15')
ax3a.set_xlabel('Purity')
ax3a.set_ylabel('Total shots')
ax3a.set_yscale('log')
ax3a.set_xlim(0,1)
ax3a.set_ylim(3,max(y))
fig3a.colorbar(cf, ax=ax3a, label='Probability')
fig3a.tight_layout()

fig3b, ax3b = plt.subplots(figsize=(16,10))
data = np.genfromtxt(fr'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\QST_data\eigen_3D_{n}_med_tol.csv',delimiter=',')
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
cf = ax3b.contourf(X ,
                Y , Z, levels=levels,
                cmap=cmap)
ax3b.set_title(f'2D heatmap of the probability of QST giving a 0 eigenenergy - {n} Qubits - Medium zero tolerance: 1e-10')
ax3b.set_xlabel('Purity')
ax3b.set_ylabel('Total shots')
ax3b.set_yscale('log')
ax3b.set_xlim(0,1)
ax3b.set_ylim(3,max(y))
fig3b.colorbar(cf, ax=ax3b, label='Probability')
fig3b.tight_layout()

fig3c, ax3c = plt.subplots(figsize=(16,10))
data = np.genfromtxt(fr'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\QST_data\eigen_3D_{n}_low_tol.csv',delimiter=',')
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
cf = ax3c.contourf(X ,
                Y , Z, levels=levels,
                cmap=cmap)
ax3c.set_title(f'2D heatmap of the probability of QST giving a 0 eigenenergy - {n} Qubits - Low zero tolerance: 1e-5')
ax3c.set_xlabel('Purity')
ax3c.set_ylabel('Total shots')
ax3c.set_yscale('log')
ax3c.set_xlim(0,1)
ax3c.set_ylim(3,max(y))
fig3c.colorbar(cf, ax=ax3c, label='Probability')
fig3c.tight_layout()

fig4a, ax4a = plt.subplots(figsize=(16,10))
data = np.genfromtxt(fr'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\QST_data\eigen_3D_{n}_high_tol.csv',delimiter=',')
x=data[:,0]
y=data[:,1]/(3**n)
z=data[:,2]
x=np.unique(x)
y=np.unique(y)
X,Y = np.meshgrid(x,y)
Z=z.reshape(len(y),len(x))
levels = MaxNLocator(nbins=50).tick_values(Z.min(), Z.max())
cmap = 'inferno'
norm = BoundaryNorm(levels, ncolors=50, clip=True)
cf = ax4a.contourf(X ,
                Y , Z, levels=levels,
                cmap=cmap)
ax4a.set_title(f'2D heatmap of the probability of QST giving a 0 eigenenergy - {n} Qubits - High zero tolerance: 1e-15')
ax4a.set_xlabel('Purity')
ax4a.set_ylabel('Shots per basis')
ax4a.set_yscale('log')
ax4a.set_xlim(0,1)
ax4a.set_ylim(1,max(y))
fig4a.colorbar(cf, ax=ax4a, label='Probability')
fig4a.tight_layout()

fig4b, ax4b = plt.subplots(figsize=(16,10))
data = np.genfromtxt(fr'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\QST_data\eigen_3D_{n}_med_tol.csv',delimiter=',')
x=data[:,0]
y=data[:,1]/(3**n)
z=data[:,2]
x=np.unique(x)
y=np.unique(y)
X,Y = np.meshgrid(x,y)
Z=z.reshape(len(y),len(x))
levels = MaxNLocator(nbins=50).tick_values(Z.min(), Z.max())
cmap = 'inferno'
norm = BoundaryNorm(levels, ncolors=50, clip=True)
cf = ax4b.contourf(X ,
                Y , Z, levels=levels,
                cmap=cmap)
ax4b.set_title(f'2D heatmap of the probability of QST giving a 0 eigenenergy - {n} Qubits - Medium zero tolerance: 1e-10')
ax4b.set_xlabel('Purity')
ax4b.set_ylabel('Shots per basis')
ax4b.set_yscale('log')
ax4b.set_xlim(0,1)
ax4b.set_ylim(1,max(y))
fig4b.colorbar(cf, ax=ax4b, label='Probability')
fig4b.tight_layout()

fig4c, ax4c = plt.subplots(figsize=(16,10))
data = np.genfromtxt(fr'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\QST_data\eigen_3D_{n}_low_tol.csv',delimiter=',')
x=data[:,0]
y=data[:,1]/(3**n)
z=data[:,2]
x=np.unique(x)
y=np.unique(y)
X,Y = np.meshgrid(x,y)
Z=z.reshape(len(y),len(x))
levels = MaxNLocator(nbins=50).tick_values(Z.min(), Z.max())
cmap = 'inferno'
norm = BoundaryNorm(levels, ncolors=50, clip=True)
cf = ax4c.contourf(X ,
                Y , Z, levels=levels,
                cmap=cmap)
ax4c.set_title(f'2D heatmap of the probability of QST giving a 0 eigenenergy - {n} Qubits - Low zero tolerance: 1e-5')
ax4c.set_xlabel('Purity')
ax4c.set_ylabel('Shots per basis')
ax4c.set_yscale('log')
ax4c.set_xlim(0,1)
ax4c.set_ylim(1,max(y))
fig4c.colorbar(cf, ax=ax4c, label='Probability')
fig4c.tight_layout()

#%%
