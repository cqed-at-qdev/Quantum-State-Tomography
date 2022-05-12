#%%
from qutip import *
from qutip.qip.operations import snot
import numpy as np
import pandas as pd
def invert_nhadamards(num):
    df_old = pd.read_pickle("Inverted_hadamards.pkl")
    n = np.arange(1,num)
    if n == np.arange(1, len(df_old[0])):
    matrix_list = []
    for i in n:
        num_hadamard = [snot()]*i
        hadamard = tensor(num_hadamard)*(1/np.sqrt(2))**i
        inv_had = hadamard.inv()
        matrix_list.append(inv_had)
    print(matrix_list)
    idx = n
    df = pd.DataFrame(data=matrix_list, index = [str(x) for x in idx])
    df.to_pickle("Inverted_hadamards.pkl")

#%%
result = [df[0][ind] for ind in df.index]
print(result[0])
#%%
print(len(df[0]))