#%%
import numpy as np
from qutip import *
from Linear_inversion import LinearInversion as LI
from QST_nqubit import QStateTomo as QST
from Data_sim import DataSimNqubit as DS
import matplotlib.pyplot as plt
import pandas as pd
import csv
import settings
#%%
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
def is_rho_phys(rho):
    eigen = rho.eigenenergies()
    if any(i < 0 for i in eigen)==True:
        return False
    else:
        return True
# %%
qubit = 1
paulis, expect_val = pauli_calculator(qubit)
hadamard = pd.read_pickle("Inverted_hadamards.pkl")[0][qubit-1]
expect_key = list(filter(None, expect_val.values.tolist()[qubit-1]))
x = False
trace_array = np.zeros(shape=(20,300))
for k in range(300):
    while x==False:
        data = DS(1, 100, tomo_type='MLE', explicit=False, had_exp=hadamard, expect_key=expect_key)
        expect_dict, expect = data.measure_qubit()
        del expect[0]
        LI_tomo = LI(1, expect, explicit=False, expect_key=expect_key)
        rho_li = LI_tomo.get_lin_inv_rho()
        x = is_rho_phys(rho_li)

    QS_tomo = QST(expect, 1, explicit=False, paulis=paulis)
    rho_qst = QS_tomo.getRho()
    trace_list = [tracedist(i, rho_li) for i in settings.rho_MLE]
    trace_array[k,:] = trace_list[:20]
trace_mean = np.mean(trace_array, axis=1)
f = open(fr'C:\Users\caspe\OneDrive\Desktop\PUK - Morten\Code\Quantum-State-Tomography\trace_dist_QST_LI_tole-5', 'a', newline='')
writer = csv.writer(f)
f.close()