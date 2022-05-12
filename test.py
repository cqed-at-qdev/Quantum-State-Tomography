#%%
# Packages
import qutip
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from Data_generation import Datagen
from QST_1qubit import QStateTomo
import settings as s
s.init()
import Functions as f
#%%
datagen = Datagen(1000, a=1+np.sqrt(2), b=1)
Paulis = [sigmax(), sigmay(), sigmaz()]
rho_true = datagen.get_density()
distance = []
for i in range(100):
    rho_list = []
    base = np.random.choice(['x','y','z'], 1000)
    for i in base:
        base0, base1, proj0, proj1 = datagen.get_basis(i)
        P0 = (proj0*rho_true).tr()
        P1 = (proj1*rho_true).tr()
        measur = np.random.choice([1,-1], 1, p=[P0,P1])
        if measur == 1:
            state = base0
        if measur == -1:
            state = base1
        rho_i = 3*state*state.conj().trans() - qeye(2)
        rho_list.append(rho_i)
    rho_MLE = (sum(rho_list)/len(rho_list)).unit()
    d = tracedist(rho_true, rho_MLE)
    distance.append(d)
np.mean(distance)
#%%
datagen = Datagen(10000, a=1+np.sqrt(2), b=1)
Paulis = [sigmax(), sigmay(), sigmaz()]
rho_true = datagen.get_density()
distance = []
for x in range(10):
    rho_list = []
    base0 = [rand_ket(2,1) for _ in range(1000)]
    base0 = [(x*x.conj().trans()).unit() for x in base0]
    base1 = [qeye(2)-x for x in base0]
    for i in range(len(base0)):
        P0 = (base0[i]*rho_true).tr()
        P1 = (base1[i]*rho_true).tr()
        measur = np.random.choice([1,-1], 1, p=[P0,P1])
        if measur == 1:
            state = base0[i]
        if measur == -1:
            state = base1[i]
        rho_i = 3*state - qeye(2)
        rho_list.append(rho_i)
    rho_MLE = (sum(rho_list)/len(rho_list)).unit()
    d = tracedist(rho_true, rho_MLE)
    distance.append(d)
random = np.mean(distance)
print(random)
#%%
datagen = Datagen(1000, a=1+np.sqrt(2), b=1)
Paulis = [sigmax(), sigmay(), sigmaz()]
rho_true = datagen.get_density()
distance = []
for x in range(100):
    datafile = datagen.getdatafile()
    Tomo = QStateTomo(datafile, Paulis, 1)
    rho1 = Tomo.getRho()
    d = tracedist(rho1, rho_true)
    distance.append(d)
print(np.mean(distance))