#%%
import numpy as np
import qutip
from qutip import *
import Functions_nqubits as f
import settings as s
import pandas as pd
class QStateTomo():
    """Class that implements 1 qubit Quantum State Tomography
    """
    def __init__(self, datafile, nqubits, explicit=True, paulis=0):
        """Initzialise function of the QST class.

        Args:
            datafile ([list]): Takes a list of the three expectation values of the state.
            Paulis ([list]): Takes the three pauli matrices as input
            nqubits ([integer]): Number of qubits to do tomography on.
        """
        self.datafile = datafile
        self.nqubits = nqubits
        s.rho_anim_list = []
        if explicit == True:
            self.Paulis = self.pauli_calculator()
        else:
            self.Paulis = paulis
        
    def getRho(self):
        """Function that estimates the density matrix rho using MLE.

        Returns:
            [Qobj/array]: The best estimate of rho in the format of a Qutip quantum object.
        """
        t = f.MLE_QST(self.datafile,self.Paulis,self.nqubits)
        rho = f.op_cholesky(t)
        return Qobj(rho,dims=[[2]*self.nqubits,[2]*self.nqubits])
    
    def pauli_calculator(self):
        """Takes the row from the datasim_keylist corresponding to n qubits
        and converts it to a set of pauli matrices, which get tensored.
        Ex. if the current key is 'x2x3' it turns into 'tensor(I,sig_x,sig_x).

        Returns:
            Paulis: ([list]): returns list of pauli tensor products
        """
        # Read pickle file into dataframed
        df1 = pd.read_pickle("datasim_keylist.pkl")
        # Remove any None values and grab the correct row
        expect_key = list(filter(None, df1.values[self.nqubits-1]))
        # Remove duplicates
        expect_key = list(dict.fromkeys(expect_key))
        paulis = []
        # Loop through every key in the list 
        for key in expect_key:
            tup_list = []
            num_array= np.zeros(self.nqubits)
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
            if len(tup_list)==self.nqubits:
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
        return paulis