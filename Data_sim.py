import numpy as np
from qutip import *
from qutip.measurement import measure, measurement_statistics,measurement_statistics_povm
from qutip.qip.operations import snot
import itertools as it
from collections import Counter
import pandas as pd
import pickle5 as pickle
class DataSimNqubit():
    """Class that can simulate data for nqubits. 
    """

    def __init__(self, nqubits, shots, prob_amp = [], prob_angles = [], basis_angles = [np.pi/2, 0, -np.pi/2, 0, np.pi/2, np.pi/2, np.pi/2, -np.pi/2, 0, 0, np.pi, 0], entangle=True, purity=1, verbose=False, explicit=True, had_exp=0, tomo_type='MLE', expect_key=0, total_shot=False):
        """Inititialize the DataSimNqubit class. 
        Input the number of qubits to simulate and the number of shots which need to be simulated
        for each measuring basis.
        Input a list of probability amplitudes (2 for each qubit).
        If no probability amplitudes are given, a random quantum state is generated.
        Input a list of measuring basis (a thetha and phi value for each of the three basis)
        Deafaults to xyz basis.
        Args:
            nqubits ([int]): how many qubits to simulate
            shots ([int]): how many shots to measure in each basis
            prob_amp ([list]): list of probability amplitudes for the combined quantum state, two for each qubit
            basis_angles ([list]): list of angles to be used to create the measuring basis.
            entangle ([bool]): Determines whether the random state created are entangled or not.
        """
        self.nqubits = nqubits
        self.shots = shots
        self.dims = 2**nqubits
        self.entangle = entangle
        self.basis_angles = basis_angles
        self.verbose = verbose
        self.explicit = explicit
        self.tomo_type = tomo_type
        self.total_shot = total_shot
        if purity < 1/self.dims:
            print('ERROR! Purity can not be smaller than 1/(2^n)')
        else:
            self.epsilon = 1 - np.sqrt(1-((1-purity)/(1-(1/self.dims))))
        # Initialize random qubits
        if len(prob_amp) == 0:
            if len(prob_angles)==0:
                self.qubit = self.init_rand_nqubits()
            else:
                self.prob_amp = 0
                self.prob_angles = prob_angles
                self.qubit = self.init_nqubits()
        
        # Initialize given qubits
        if len(prob_amp) == 2*nqubits:
            self.prob_amp = prob_amp
            self.qubit = self.init_nqubits()
        # Not enough probability amplitudes
        if 0 < len(prob_amp) < 2*nqubits or 2*nqubits < len(prob_amp):
            print('ERROR: Number of probability amplitudes dont match the number of qubits')
        # Print the true density matrix
        self.rho_true = self.get_density()
        if self.verbose == True:
            print(f"The true density matrix is: {self.rho_true}")
            print('------------------------------------------')
        # Read the correct inverted Hadamard matrix
        if self.explicit==True:
            with open(r"Inverted_hadamards.pkl", "rb") as fh:
                self.df = pickle.load(fh)
            if self.nqubits <= len(self.df[0]):
                self.hadamard = self.df[0][self.nqubits-1]
            else:
                num_hadamard = [snot()]*self.nqubits
                hadamard = tensor(num_hadamard)*(1/np.sqrt(2))**self.nqubits
                self.hadamard = hadamard.inv()
        else:
            self.hadamard = had_exp
        
        # Read in the correct set of keys for the expectations values for n qubits
        if self.explicit==True:
            with open(r"datasim_keylist.pkl", "rb") as fh:
                self.df1 = pickle.load(fh)
            expect_key = list(filter(None, self.df1.values[self.nqubits-1]))
        else:
            pass
    
        self.expect_key = expect_key
    def init_basis(self):
        """Initializes the measuring basis. Currently only xyz

        Returns:
            basis_set ([list]): List of all basis vectors.
            proj_dict ([dict]): Dictionary containing all needed projectors (two for each basis).

        """
        # Create basis
        angle_set = [self.basis_angles[i * 2:(i + 1) * 2] for i in range((len(self.basis_angles) + 2 - 1) // 2 )] 
        basis_set = [(np.cos(x[0]/2)*basis(2,0)+np.exp(1j*x[1])*np.sin(x[0]/2)*basis(2,1)).unit() for x in angle_set]
        # Calculate corresponding projectors
        proj_set = [base.proj() for base in basis_set]
        proj_dict = {
            'x0': proj_set[0],
            'x1': proj_set[1],
            'y0': proj_set[2],
            'y1': proj_set[3],
            'z0': proj_set[4],
            'z1': proj_set[5]
        }
        return basis_set, proj_dict
    
    def init_rand_nqubits(self):
        """Initializes random qubits. Can create pure or mixed states.

        Returns:
            qubit ([Qobj]): returns the tensor product of all the qubits.
        """
        # Create a unentangled quantum state
        if self.entangle == False:
            qubit = [rand_ket(2) for _ in range(self.nqubits)]
            qubit = (tensor(qubit)).proj()
            qubit = (1-self.epsilon)*qubit + self.epsilon*(1/self.dims)*qeye([2]*self.nqubits)
        # Create a entangled quantum state
        if self.entangle == True:
            qubit = rand_ket(2**self.nqubits, dims=[[2]*self.nqubits,[1]*self.nqubits])
            qubit = qubit.proj()
            qubit = (1-self.epsilon)*qubit + self.epsilon*(1/self.dims)*qeye([2]*self.nqubits)
        # Calculate purity of qubits
        if self.verbose == True:
            for i in range(self.nqubits):
                partial_trace = ptrace(qubit,i)
                purity = (partial_trace**2).tr()
                print(f"purity of qubit{i} = {purity} - {partial_trace}")
                print('------------------------------------------')
            # Calculate purity of system
            purity = (qubit*qubit).tr()
            print(f"purity of system = {purity}")
            print('------------------------------------------')
        return qubit

    def init_nqubits(self):
        """Generate qubits from given list of probability amplitudes.

        Returns:
            qubit ([Qobj]): returns the tensor product of all the qubits.
        """
        # Split the amplitude list into sets of two, two for each qubit.
        if self.prob_amp == 0:
            ang_set = [self.prob_angles[i * 2:(i + 1) * 2] for i in range((len(self.prob_angles) + 2 - 1) // 2 )] 
            qubit_list = [(np.cos(x[1]/2)*basis(2,0)+np.exp(1j*x[0])*np.sin(x[1]/2)*basis(2,1)).unit() for x in ang_set]        
            qubit = (tensor(qubit_list)).proj()
            qubit = (1-self.epsilon)*qubit + self.epsilon*(1/self.dims)*qeye([2]*self.nqubits)
        else:
            amp_set = [self.prob_amp[i * 2:(i + 1) * 2] for i in range((len(self.prob_amp) + 2 - 1) // 2 )] 
            qubit_list = [(x[0]*basis(2,0)+x[1]*basis(2,1)).unit() for x in amp_set]        
            qubit = (tensor(qubit_list)).proj()
        qubit = (1-self.epsilon)*qubit + self.epsilon*(1/self.dims)*qeye([2]*self.nqubits)
        # Calculate purity of qubits
        if self.verbose == True:
            for i in range(self.nqubits):
                partial_trace = ptrace(qubit,i)
                purity = (partial_trace**2).tr()
                print(f"purity of qubit{i} = {purity} - {partial_trace}")
                print('------------------------------------------')
            # Calculate purity of system
            purity = (qubit*qubit).tr()
            print(f"purity of system = {purity}")
            print('------------------------------------------')
        return qubit

    def get_density(self):
        """Calculates the true density matrix of the state.

        Returns:
            rho_true ([Qobj]): The true density matrix as a quantum object
        """
        rho_true = self.qubit
        return rho_true

    def calc_prob_nsite(self):
        """This function calculates the probability distributions between all outcomes for all twosite measurements. 
        Twosite meaning all possible twoqubit correlations.

        Returns:
            prob_dict ([dict]): A dictionary with all possible two qubit
        """
        # Initialize basis and projectors
        basis_set, proj_dict = self.init_basis()

        # Create a list of all possible correlation posibilities for n qubits
        columns = list(it.product(['x','y','z'], repeat=self.nqubits))
        # Create a list of all possible outcomes for n qubits
        row = list(it.product(['0','1'], repeat=self.nqubits))
        # Merge into a matrix
        proj_matrix = []
        for i in columns:
            for a in row:
                proj_matrix.append([k + g for k,g in zip(i,a)])
        proj_matrix = np.array(proj_matrix,ndmin = 3).reshape(len(columns),-1,len(row[0]))
        # Translate the strings of projectors into Qobj and tensor them together
        tens_mat = []
        for a in proj_matrix:
            for b in a:
                x = tensor(list(map(proj_dict.__getitem__,b)))
                tens_mat.append(x)
        # Feed the set of projectors to qutip's measurement function, 
        # which outputs the probability distribution of the possible outcomes.
        n = 2**self.nqubits
        final = [tens_mat[i * n:(i + 1) * n] for i in range((len(tens_mat) + n - 1) // n )] 
        prob = [measurement_statistics_povm(self.qubit, i)[-1] for i in final]
        # Create a dictionary of probability distributions with 
        prob_dict = {
            columns[i]: prob[i] for i in range(len(columns))
            }
        return prob_dict, tens_mat

    def measure_qubit(self):
        """Function which takes the probability distributions and measures the qubits
        for n shots in each of the correlated basis.

        Returns:
            expect_dict ([dict]): A dictionary containing all needed measurments.
        """
        # Calculate the probability distributions
        prob_dict, tens_mat = self.calc_prob_nsite()
        # Create all possible outcomes
        row = list(it.product([-1,1], repeat=self.nqubits))
        idx = np.arange(len(row))
        expect_list = []
        count_list = []
        # Loop through all sets of probabilties and choose between 
        # the different outcomes with the given probabilities
        if self.total_shot == False:
            pass
        elif self.total_shot == True:
            self.shots = int(self.shots/(3**self.nqubits))
        for i, prob in enumerate(prob_dict.values()):
            # Measure for n shots
            measure = list(np.random.choice(len(row), self.shots, p=prob))
            # Count the frequency of each outcome and find the measured probability
            counts = np.array([measure.count(x)/self.shots for x in idx])
            count = counts*self.shots
            count_list.append(count)
            # Use this to calculate all expectation values
            expect1 = (self.hadamard*counts).real
            # Remove identity
            expect1 = np.delete(expect1, 0)
            # Append to 
            expect_list.extend(expect1)
        # Create dictionary to hold the expectation values
        expect_dict = {
            'I': 1.0
        }
        for key, value in zip(self.expect_key, expect_list):
            if key in expect_dict:
                expect_dict[key].append(value)
            else:
                expect_dict[key] = [value]
        # Average out all values with the same key.
        for key, value in zip(self.expect_key, expect_list):
            value = np.mean(value)
            expect_dict[key] = value
        if self.verbose == True:
            print(f"Expected number of expectation values = {4**self.nqubits}","\n",f"Actual number of expectation values = {len(expect_dict)}")
        expect_values = list(expect_dict.values())
        if self.tomo_type == 'MLE':
            return expect_dict, expect_values
        if self.tomo_type == 'Bayesian':
            return [x for l in count_list for x in l], tens_mat
        if self.tomo_type == 'All':
            return expect_dict, expect_values, [x for l in count_list for x in l], tens_mat