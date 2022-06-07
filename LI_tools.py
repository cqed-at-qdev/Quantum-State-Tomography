from qutip import *
import numpy as np
def pauli_calculator(nqubits, expect_key):
    """Takes the row from the datasim_keylist corresponding to n qubits
    and converts it to a set of pauli matrices, which get tensored.
    Ex. if the current key is 'x2x3' it turns into 'tensor(I,sig_x,sig_x).

    Returns:
        Paulis: ([list]): returns list of pauli tensor products
    """
    paulis = [qeye([2]*nqubits)]
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
    paulis = np.array(paulis)
    return paulis