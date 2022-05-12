import numpy as np
import qutip
from qutip import *
import Functions as f
import settings as s
class QStateTomo():
    """Class that implements 1 qubit Quantum State Tomography
    """
    def __init__(self, datafile, Paulis, nqubits=1):
        """Initzialise function of the QST class.

        Args:
            datafile ([list]): Takes a list of the three expectation values of the state.
            Paulis ([list]): Takes the three pauli matrices as input
            nqubits ([integer]): Number of qubits to do tomography on.
        """
        self.datafile = datafile
        self.Paulis = Paulis
        self.nqubits = nqubits
        s.rho_anim_list = []
        
        
    def getRho(self):
        """Function that estimates the density matrix rho using MLE.

        Returns:
            [Qobj/array]: The best estimate of rho in the format of a Qutip quantum object.
        """
        t = f.MLE_QST(self.datafile,self.Paulis,self.nqubits, self.datafile)
        rho = f.op_cholesky(t)
        return Qobj(rho)