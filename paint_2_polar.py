#%%
import matplotlib.pyplot as plt
from matplotlib import markers, pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
from colormap import rgb2hex
from qutip import *
%matplotlib widget
#%%
def array_2_polarrad(x,y, res):
    phi = np.deg2rad(360*x/res[0])
    theta = 2*np.arctan(np.exp(np.deg2rad(180*y/res[1])))-(np.pi/2)
    return phi, theta

def png_2_polar(path, resolution=[100,50]):
    image = Image.open(path).resize(resolution,Image.ANTIALIAS)
    res = image.size
    image_bw = image.convert('L')
    image_rgb = image.convert('RGB')
    image_bw_array = np.array(image_bw)
    y_pre,x_pre = np.where(image_bw_array!=255)
    rgb = [rgb2hex(*image_rgb.getpixel((i,j))) for i,j in zip(x_pre,y_pre)]
    x = x_pre-(res[0]/2)
    y = y_pre-(res[1]/2)
    phi, theta = array_2_polarrad(x,y,res)
    return phi, theta, rgb

def plot_polar_bloch(phi, theta, rgb='k'):
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    fig = pyplot.figure()
    ax = Axes3D(fig)
    b = Bloch(axes=ax)
    b.point_color = rgb
    for i in range(len(x)):
        b.add_points([x[i],y[i],z[i]])
    b.show()

    return x,y,z
#%%
path = r'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\smiley.png'
phi, theta, rgb = png_2_polar(path)
x,y,z = plot_polar_bloch(phi,theta,rgb)
#%%
#!/usr/bin/env python3
import numpy as np
from copy import copy
from sequence import Sequence
import gates
class CustomSequence(Sequence):
    def generate_sequence(self, config, phi, theta):
        """
        CS for generating single-qubit cardinal states
        """
        index = int(config.get('Parameter #1'))
        qubit_num = int(config.get('Parameter #2'))
        for i in range(len(phi)):
            if index == i:
                self.add_gate(qubit_num, gates.SingleQubitXYRotation(phi=0, theta = float(theta[i])))
                self.add_gate(qubit_num, gates.VirtualZGate(float(phi[i])))
if __name__ == '__main__':
    pass