#%%
import numpy as np
from qutip import *
import pandas as pd
import LI_tools as li
class LinearInversion():

    def __init__(self, nqubits, expectation, explicit=True, expect_key=0):
        """Initializes linear inversion tomography. Calculates the density matrix by finding the dot product between the
        the measured expectation values and the corresponding pauli matrix vector.

        Args:
            nqubits ([int]): number of qubits
            expectation (list): list of expectation values
            explicit (bool, optional): If True the datasim_keylist is read inside the class, if False
            one needs to read it externally and input it as a parameter. If tomography has to be run many times it is,
            advisable to read it externally due to it being time consuming to due each iteration. Defaults to True.
            expect_key (int, optional): The input from the pickle file. Defaults to 0.
        """
        self.nqubits = nqubits
        self.expectation = expectation
        self.explicit = explicit

        # Read pickle file into dataframe
        if self.explicit==True:
            self.df1 = pd.read_pickle("datasim_keylist.pkl")
            # Remove any None values and grab the correct row
            expect_key = list(filter(None, self.df1.values.tolist()[self.nqubits-1]))
        else:
            pass

        # Remove duplicates
        self.expect_key = list(dict.fromkeys(expect_key))
        self.paulis = li.pauli_calculator(self.nqubits, self.expect_key)

    def get_lin_inv_rho(self):
        """Calculates the linear inverted density matrix.

        Returns:
            rho ([Qobj]): returns the LI density matrix.
        """
        # np.einsum is used to calculate the dot product as fast as possible.
        rho = (1/(2**self.nqubits))*np.einsum('i,ijk->jk',np.array(self.expectation),self.paulis)
        rho = Qobj(rho, dims=[[2]*self.nqubits,[2]*self.nqubits])
        return rho