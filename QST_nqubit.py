#%%
import numpy as np
import qutip
from qutip import *
import Functions_nqubits as f
import settings as s
import pandas as pd
class QStateTomo():
    """Class that implements n qubit Quantum State Tomography
    """
    def __init__(self, datafile, explicit=True, paulis=0):
        """Initzialise function of the QST class.

        Args:
            datafile ([list]): Takes a list of the three expectation values of the state.
            Paulis ([list]): Takes the three pauli matrices as input
            nqubits ([integer]): Number of qubits to do tomography on.
        """
        self.datafile = datafile
        self.nqubits = int(np.log(len(datafile))/np.log(4))
        s.rho_anim_list = []
        if explicit == True:
            self.Paulis = f.pauli_calculator()
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
    
