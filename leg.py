#%%
import numpy as np
from qutip import *
from qutip.measurement import measure, measurement_statistics,measurement_statistics_povm

qubit = rand_ket(2).proj()
print(qubit.eigenenergies())