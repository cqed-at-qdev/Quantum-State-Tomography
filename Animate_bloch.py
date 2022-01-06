## Work in progress
import qutip
from qutip import *
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import settings
class AnimateBlochSphere():
    def __init__(self):
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig, azim=-40, elev=30, )
        self.sphere = Bloch(axes=self.ax)
        self.sphere.vector_color = ['g', 'r', 'b']
        self.ax.legend()

    def animate(self, i):
        self.sphere.clear()
        self.sphere.add_states(settings.rho_true)
        self.sphere.add_states(settings.rho_basic)
        self.sphere.add_states(settings.rho_anim_list[i])
        self.sphere.make_sphere()
        return self.ax

    def anim(self):
        ani = animation.FuncAnimation(self.fig, self.animate(), np.arange(len(settings.rho_anim_list)),
                                    init_func=self.__init__(), blit=False, repeat=False)
        ani.save(r'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\{bloch_sphere2.gif}', fps=2)