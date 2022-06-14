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
from QST_nqubit import QStateTomo
from Data_sim import DataSimNqubit
import settings as s
s.init()
import Functions_nqubits as f
#%%
Nqubits = 1

State1 = DataSimNqubit(Nqubits, 1000)
dictionary, data = State1.measure_qubit()
s.rho_true = State1.get_density()
del data[0]
#%%
print(s.rho_true)
#%%
Tomo1 = QStateTomo(data, Nqubits)
rho1 = Tomo1.getRho()
#%%
print(tracedist(s.rho_true,rho1))
#%%
plt.plot(np.arange(len(s.tracedist_list)),s.tracedist_list)
#%%
fig = plt.figure(figsize=(16,16))
ax = Axes3D(fig, azim=-40, elev=30, )
sphere = Bloch(axes=ax)

def animate(i):
   sphere.clear()
   sphere.add_states(s.rho_true)
   sphere.add_states(s.rho_basic)
   sphere.add_states(s.rho_anim_list[i])
   sphere.make_sphere()
   g_patch = mpatches.Patch(color='g', label='True state')
   r_patch = mpatches.Patch(color='r', label='Basic state')
   b_patch = mpatches.Patch(color='b', label='MLE state')

   leg = ax.legend(handles=[g_patch, r_patch, b_patch], fontsize=18)
   return ax, leg

def init():
   sphere.vector_color = ['g', 'r', 'b']

   return ax

ani = animation.FuncAnimation(fig, animate, np.arange(len(s.rho_anim_list)),
                              init_func=init, blit=False, repeat=False)
ani.save(r'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\bloch_sphere_kardinal.gif', fps=2)
#%%
plt.plot(s.tracedist_list)
plt.xlabel('MLE step')
plt.ylabel('Trace distance')
print(s.tracedist_list)
#%%
#%%
rho2 = -(0.5 * (qeye(2)+data[0]*sigmax()+data[1]*sigmay()+data[2]*sigmaz()))
print(rho2)
#%%
b = Bloch()
b.add_states(rho_true)
b.add_states(rho2)
b.add_states(rho1)
b.show()
print(data)
print(rho_true)
#%%
blo = Bloch()
blo.add_states(basis(2,0))
blo.show()