#%%
# Packages
import qutip
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from Data_generation import Datagen
from QST_1qubit import QStateTomo
import settings as s
s.init()
import Functions as f
#%%
a = 1/np.sqrt(2)
b = np.exp(1j*np.pi/8)
State1 = Datagen(a,b,1000)
datafile = State1.getdatafile()
rho_true = State1.get_density()
Paulis = [sigmax(), sigmay(), sigmaz()]
#%%
Tomo1 = QStateTomo(datafile, Paulis, 1)
rho1 = Tomo1.getRho()
print(rho1, rho_true)
#%%
fig = plt.figure()
ax = Axes3D(fig, azim=-40, elev=30, )
sphere = Bloch(axes=ax)

def animate(i):
   sphere.clear()
   sphere.add_states(rho_true)
   sphere.add_states(s.rho_basic)
   sphere.add_states(s.rho_anim_list[i])
   sphere.make_sphere()
   return ax

def init():
   sphere.vector_color = ['g', 'r', 'b']
   ax.legend()
   return ax

ani = animation.FuncAnimation(fig, animate, np.arange(len(s.rho_anim_list)),
                              init_func=init, blit=False, repeat=False)
ani.save(r'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\bloch_sphere2.gif', fps=2)
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
print(qeye(2))