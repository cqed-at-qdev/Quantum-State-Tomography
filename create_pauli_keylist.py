#%%
import numpy as np
from qutip import *
import pandas as pd


def pauli_calculator(n):
        df1 = pd.read_pickle("datasim_keylist.pkl")
        expect_key = list(filter(None, df1.values.tolist()[n-1]))
        expect_key = list(dict.fromkeys(expect_key))
        paulis = []
        for key in expect_key:
            tup_list = []
            num_array= np.zeros(n)
            for value in key:
                if 'x' in value:
                    tup_list.append(sigmax())
                if 'y' in value:
                    tup_list.append(sigmay())
                if 'z' in value:
                    tup_list.append(sigmaz())
                if value != 'x':
                    if value != 'y':
                        if value != 'z':
                            num_array[int(value)-1] = value
                else:
                    pass
            if len(tup_list)==n:
                pauli_set = tensor(tup_list)
                paulis.append(pauli_set)
            else:
                for i,x in enumerate(num_array):
                    if x==0:
                        tup_list.insert(i,qeye(2))
                    else:
                        pass
                pauli_set = tensor(tup_list)
                paulis.append(pauli_set)
        return paulis
#%%
n_loop = np.arange(1,10)
pauli_list = []
for n in n_loop:
    x = pauli_calculator(n)
    pauli_list.append(x)
df = pd.DataFrame(data=pauli_list, index = [str(x) for x in n_loop])
df.to_pickle("pauli_keylist.pkl")
#%%
df1 = pd.read_pickle("pauli_keylist.pkl")
expect_key = list(filter(None, df1.values.tolist()[6-1]))
print(expect_key)
print(len(expect_key))