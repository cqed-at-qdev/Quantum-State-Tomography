#%%
import csv
import numpy as np
import itertools as it
import pandas as pd
nqubits = np.arange(1,10)
key_master = []
for n in nqubits:
    columns = list(it.product(['x','y','z'], repeat=n))
    # Create a list of all possible outcomes for n qubits
    row = list(it.product(['0','1'], repeat=n))
    # Merge into a matrix
    proj_matrix = []
    for i in columns:
        for a in row:
            proj_matrix.append([k + g for k,g in zip(i,a)])
    proj_matrix = np.array(proj_matrix,ndmin = 3).reshape(len(columns),-1,len(row[0]))
    key_list = []
    for j in proj_matrix:
        for o,i in enumerate(j):
            for k in range(len(i)):
                if i[k] == 'x1':
                    i[k] = 'x' + str(k+1)
                if i[k] == 'y1':
                    i[k] = 'y' + str(k+1)
                if i[k] == 'z1':
                    i[k] = 'z' + str(k+1)
                if i[k] == 'x0':
                    i[k] = 0
                if i[k] == 'y0':
                    i[k] = 0
                if i[k] == 'z0':
                    i[k] = 0
            i = i[i != '0']
            #i = ''.join(i)
            key_list.append(i)
    #key_list = [i for i in key_list if i]
    key_master.append(key_list)
df = pd.DataFrame(data=key_master, index = [str(x) for x in nqubits])
df.to_pickle("pauli_keylist.pkl")