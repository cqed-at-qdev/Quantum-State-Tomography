#%%
import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from Linear_inversion import LinearInversion as LI
import LI_tools as li_tools
from QST_nqubit import QStateTomo as MLE
from Bayesian_tomo import BayInfTomo as BI
from Data_sim import DataSimNqubit as datasim
import Labber
import quantum_fitter.readout_tools as rdt
import pandas as pd
from matplotlib import markers, pyplot
from mpl_toolkits.mplot3d import Axes3D
from colour import Color
%matplotlib widget
#%%
def create_classifier(calib_path, size=3000):
    calib_file_str = calib_path
    calib_file = rdt.Readout(calib_file_str, size=size)
    calib_file.plot_classifier()
    calib_file.export_classifier()

def get_expec_from_lab(path,calib_pickle):
    classifier = rdt.import_classifier(calib_pickle)
    tomo_file_name = path
    tomo_file = Labber.LogFile(tomo_file_name)
    tomo_data = tomo_file.getData()
    x = tomo_data.real
    y = tomo_data.imag
    expect_master = np.array([(np.mean(classifier.predict(np.vstack([x[i],y[i]]).T))-0.5)*-2 for i in range(len(x))])
    if len(expect_master)>3:
        expect_master = np.array([[expect_master[i+2],expect_master[i+1],expect_master[i]] for i in np.arange(0,len(x),3)])
    else:
        expect_master = np.array([[expect_master[2],expect_master[1],expect_master[0]]])
    return expect_master 

def plot_pulse_charchterization(expect_values):
    fig, ax = plt.subplots(figsize = (7,5))
    x_ax = np.arange(1,22,1)
    y_ax = expect_values/expect_values[0]
    ax.plot(x_ax,y_ax, marker='o')
    ax.set_xticks(x_ax)
    tick_label = ['II','XpXp','YpYp','XpYp','YpXp','X2pI','Y2pI','X2pY2p','Y2pX2p','X2pYp','Y2pXp','XpY2p','YpX2p','X2pXp','XpX2p','Y2pYp','YpY2p','XpI','YpI','X2pX2p','Y2pY2p']
    ax.set_xticklabels(tick_label, rotation=90)
    ax.set_xlabel('Pulse Sequence')
    ax.set_ylabel('Z Expectation Value')
    ax.set_title('Pulse Sequence Charachterization')
    fig.tight_layout()

def plot2bloch(expect_value, bloch_angle=None, color_range=('red','green'), markers='o', interactive=False):
    color1 = Color(color_range[0])
    colors = list(color1.range_to(Color(color_range[1]),int(len(expect_value))))
    colors = [str(i) for i in colors]
    if interactive == True:
        fig = pyplot.figure()
        ax = Axes3D(fig)
        b = Bloch(axes=ax)
    if interactive == False:
        b = Bloch()
    if interactive == False and bloch_angle != None:
        b.view = bloch_angle

    b.point_color = colors
    b.point_marker = markers

    for i, state in enumerate(expect_value):
        if type(state)!=type(Qobj()):
            b.add_points(state)
            if i<(len(expect_value)/3)-1:
                b.add_line(expect_value[i], expect_value[i+1], fmt='--', color='grey')
            else:
                pass
        else:
            x = 2*state[1,0].real
            y = 2*state[1,0].imag
            z = 2*state[0,0]-1
            b.add_points([x,y,z])
            
    b.show()

def LI_1qubit(expect_value):
    df1 = pd.read_pickle("datasim_keylist.pkl")
    expect_key = list(filter(None, df1.values.tolist()[1-1]))
    expect_key = list(dict.fromkeys(expect_key))
    paulis = li_tools.pauli_calculator(1,expect_key)
    rho_list = []
    for i in range(len(expect_value)):
        expect = [1, *expect_value[i]]
        li_tomo = LI(1,expect,explicit=False, expect_key=expect_key, paulis=paulis)
        rho = li_tomo.get_lin_inv_rho()
        rho_list.append(rho)
    return rho_list

def MLE_1qubit(expect_value):
    df1 = pd.read_pickle("datasim_keylist.pkl")
    expect_key = list(filter(None, df1.values.tolist()[1-1]))
    expect_key = list(dict.fromkeys(expect_key))
    paulis = li_tools.pauli_calculator(1,expect_key)
    rho_list = []
    for i in range(len(expect_value)):
        expect = [1, *expect_value[i]]
        mle_tomo = MLE(expect,explicit=False, paulis=paulis)
        rho = mle_tomo.getRho()
        rho_list.append(rho)
    return rho_list

def BI_1qubit(counts, povm):
    rho_list = []
    for i in range(len(counts)):
        bi_tomo = BI(counts[i], povm, 1, verbose=False, samples=10000)
        rho = bi_tomo.getRho()[0]
        rho_list.append(rho)
    return rho_list

def get_counts_from_lab(path, calib_pickle):
    count_list = []
    classifier = rdt.import_classifier(calib_pickle)
    tomo_file_name = path
    tomo_file = Labber.LogFile(tomo_file_name)
    tomo_data = tomo_file.getData()
    x = tomo_data.real
    y = tomo_data.imag
    expect_master = np.array([classifier.predict(np.vstack([x[i],y[i]]).T) for i in range(len(x))])
    if len(expect_master)>3:
        expect_master = np.array([[expect_master[i+2],expect_master[i+1],expect_master[i]] for i in np.arange(0,len(x),3)])
    else:
        expect_master = np.array([[expect_master[2],expect_master[1],expect_master[0]]])
    for state in expect_master:
        X_0_count = list(state[0]).count(0)
        X_1_count = list(state[0]).count(1)
        Y_0_count = list(state[1]).count(0)
        Y_1_count = list(state[1]).count(1)
        Z_0_count = list(state[2]).count(0)
        Z_1_count = list(state[2]).count(1)
        counts = [X_0_count, X_1_count, Y_0_count, Y_1_count, Z_0_count, Z_1_count]
        count_list.append(counts)
    X_0_povm = ((basis(2,0)+basis(2,1)).unit()).proj()
    X_1_povm = ((basis(2,0)-basis(2,1)).unit()).proj()
    Y_0_povm = ((basis(2,0)+1j*basis(2,1)).unit()).proj()
    Y_1_povm = ((basis(2,0)-1j*basis(2,1)).unit()).proj()
    Z_0_povm = (basis(2,0).unit()).proj()
    Z_1_povm = (basis(2,1).unit()).proj()
    povm_list = [X_0_povm, X_1_povm, Y_0_povm, Y_1_povm, Z_0_povm, Z_1_povm,]

    return count_list, povm_list

def all_model_bloch_top(rho_list_LI, rho_list_MLE, rho_list_BI):
    xyz_LI = np.array([[2*i[1,0].real,2*i[1,0].imag,2*i[0,0]-1] for i in rho_list_LI])
    xyz_MLE = np.array([[2*i[1,0].real,2*i[1,0].imag,2*i[0,0]-1] for i in rho_list_MLE])
    xyz_BI = np.array([[2*i[1,0].real,2*i[1,0].imag,2*i[0,0]-1] for i in rho_list_BI])
    
    b = Bloch()
    color1 = Color('#006600')
    colors = list(color1.range_to(Color('#66CC00'),int(len(expect_value))))
    colors = [str(i) for i in colors]
    b.point_color = colors
    b.point_marker = ['o']
    b.view = [90,90]
    for i in xyz_LI:
        b.add_points(i)
    b.show()

    b1 = Bloch()
    color1 = Color('#000099')
    colors = list(color1.range_to(Color('#3366FF'),int(len(expect_value))))
    colors = [str(i) for i in colors]
    b1.point_color = colors
    b1.point_marker = ['^']
    b1.view = [90,90]
    for i in xyz_MLE:
        b1.add_points(i)
    b1.show()

    b2 = Bloch()
    color1 = Color('darkred')
    colors = list(color1.range_to(Color('red'),int(len(expect_value))))
    colors = [str(i) for i in colors]
    b2.point_color = colors
    b2.point_marker = ['s']
    b2.view = [90,90]
    for i in xyz_BI:
        b2.add_points(i)
    b2.show()

    xyz_LI = np.array([[2*i[1,0].real,2*i[1,0].imag,2*i[0,0]-1] for i in rho_list_LI]).T
    xyz_MLE = np.array([[2*i[1,0].real,2*i[1,0].imag,2*i[0,0]-1] for i in rho_list_MLE]).T
    xyz_BI = np.array([[2*i[1,0].real,2*i[1,0].imag,2*i[0,0]-1] for i in rho_list_BI]).T
    b3 = Bloch()
    b3.point_color = ['g','b','r']
    b3.point_marker = ['o','^','s']
    b3.view = [90,90]
    b3.point_size = [50,75,100]

    b3.add_points(xyz_LI, alpha=1)
    b3.show()

    b3.add_points(xyz_MLE, alpha=1)

    b3.show()

    b3.add_points(xyz_BI, alpha=1)
    b3.show()

def all_model_bloch_side(rho_list_LI, rho_list_MLE, rho_list_BI):
    xyz_LI = np.array([[2*i[1,0].real,2*i[1,0].imag,2*i[0,0]-1] for i in rho_list_LI])
    xyz_MLE = np.array([[2*i[1,0].real,2*i[1,0].imag,2*i[0,0]-1] for i in rho_list_MLE])
    xyz_BI = np.array([[2*i[1,0].real,2*i[1,0].imag,2*i[0,0]-1] for i in rho_list_BI])
    
    b = Bloch()
    color1 = Color('#006600')
    colors = list(color1.range_to(Color('#66CC00'),int(len(expect_value))))
    colors = [str(i) for i in colors]
    b.point_color = colors
    b.point_marker = ['o']
    for i in xyz_LI:
        b.add_points(i)
    b.show()

    b1 = Bloch()
    color1 = Color('#000099')
    colors = list(color1.range_to(Color('#3366FF'),int(len(expect_value))))
    colors = [str(i) for i in colors]
    b1.point_color = colors
    b1.point_marker = ['^']
    for i in xyz_MLE:
        b1.add_points(i)
    b1.show()

    b2 = Bloch()
    color1 = Color('darkred')
    colors = list(color1.range_to(Color('red'),int(len(expect_value))))
    colors = [str(i) for i in colors]
    b2.point_color = colors
    b2.point_marker = ['s']
    for i in xyz_BI:
        b2.add_points(i)
    b2.show()

    xyz_LI = np.array([[2*i[1,0].real,2*i[1,0].imag,2*i[0,0]-1] for i in rho_list_LI]).T
    xyz_MLE = np.array([[2*i[1,0].real,2*i[1,0].imag,2*i[0,0]-1] for i in rho_list_MLE]).T
    xyz_BI = np.array([[2*i[1,0].real,2*i[1,0].imag,2*i[0,0]-1] for i in rho_list_BI]).T
    b3 = Bloch()
    b3.point_color = ['g','b','r']
    b3.point_marker = ['o','^','s']
    b3.point_size = [80,100,15]

    b3.add_points(xyz_LI)
    b3.show()

    b3.add_points(xyz_MLE)

    b3.show()

    b3.add_points(xyz_BI)
    b3.show()
#%%
calib_path = r'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\q2_tomography_calibrate_SS.hdf5'
create_classifier(calib_path)
#%%
path = r'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\q2_tomography_smiley_face_sweep_better.hdf5'
calib_path = r'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\q2_tomography_calibrate_SS.pickle'
expect_value = get_expec_from_lab(path, calib_path)
counts, povm = get_counts_from_lab(path, calib_path)
#%%
rho_list_LI = LI_1qubit(expect_value)
rho_list_MLE = MLE_1qubit(expect_value)
#%%
rho_list_BI = BI_1qubit(counts, povm)


#%%
all_model_bloch_side(rho_list_LI, rho_list_MLE, rho_list_BI)
#%%
all_model_bloch_top(rho_list_LI, rho_list_MLE, rho_list_BI)
#%%
plot2bloch(rho_list_LI)

