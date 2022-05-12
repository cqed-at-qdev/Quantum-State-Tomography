import numpy as np
import qutip
from qutip import *
import settings as s
class Datagen():
    """Datageneration class takes a, b values as input and creates a quantum state. 
       It then "measures" n shots and calculates the expected sigma x,y and z values, which can be used for QST.
    """
    def __init__(self, n_shots, a, b, theta=0, phi=0):
    
        """Initializes the datageneration class.

        Args:
            a ([float]): Probability amplitude of ket(zero)
            b ([float]): Probability amplitude of ket(one)
            n_shots ([integer]): Number of measurments
            base (list, optional): Which base to measure in. Defaults to ['x','y','z'].
        """
        self.a = a
        self.b = b
        self.shots = n_shots
        self.theta = theta
        self.phi = phi
        self.rho_true = self.get_density()
        s.rho_true = self.get_density()
                
    def get_basis(self, base_index):
        """Generates a basis to measure in.

        Args:
            base_index ([string or float]): Indictates which basis to generate.

        Returns:
            [Qobj/array]: Returns the two basis vectors and the two corresponding projectors.
        """
        if base_index=='x':
            basis0 = (basis(2,0)+basis(2,1))/(basis(2,0)+basis(2,1)).norm()
            basis1 = (basis(2,0)-basis(2,1))/(basis(2,0)-basis(2,1)).norm()
        if base_index=='y':
            basis0 = (basis(2,0)+1j*basis(2,1))/(basis(2,0)+1j*basis(2,1)).norm()
            basis1 = (basis(2,0)-1j*basis(2,1))/(basis(2,0)-1j*basis(2,1)).norm()
        if base_index=='z':
            basis0 = (basis(2,0))/(basis(2,0)).norm()
            basis1 = (basis(2,1))/(basis(2,1)).norm()
        if base_index=='r':
            basis0 = (np.cos(self.theta/2)*basis(2,0))
            basis1 = (np.exp(1j*self.phi)*np.cos(self.theta/2)*basis(2,1))

        # Projectors
        proj0 = basis0*basis0.conj().trans()
        proj1 = basis1*basis1.conj().trans()

        return basis0, basis1, proj0, proj1

    def get_density(self):
        """Calculates the density matrix for the given a and b values.

        Returns:
            [Qobj/array]: Density matrix to generate data from.
        """
        basis0, basis1, proj1, proj2 = self.get_basis('r')
        psi = (self.a*basis0+self.b*basis1)/(self.a*basis0+self.b*basis1).norm()
        psi_com_con = psi.conj().trans()
        rho_true = psi*psi_com_con
        return rho_true

    def sig_expect(self, base_index):
        """Calculates the expectation values in the given base.

        Args:
            base_index ([string or float]): Indictates which basis to measure in.

        Returns:
            [float]: Expectation value in given basis.
        """
        base0, base1, proj0, proj1 = self.get_basis(base_index)
        P0 = (proj0*self.rho_true).tr()
        P1 = (proj1*self.rho_true).tr()
        measurment_list = np.random.choice([1,-1], self.shots, p=[P0,P1])
        sig_expect = np.mean(measurment_list)
        return sig_expect

    def getdatafile(self):
        """Generates datafile.

        Returns:
            [list]: Returns a datafile with expectation values in the chosen basis.
        """
        self.base=['x','y','z']
        sigx = self.sig_expect(self.base[0])
        sigy = self.sig_expect(self.base[1])
        sigz = self.sig_expect(self.base[2])
        datafile = [sigx, sigy, sigz]
        return datafile