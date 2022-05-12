#%%
import cProfile, pstats, io
from qutip import *
import pandas as pd
import numpy as np
from Data_sim import DataSimNqubit
from Linear_inversion import LinearInversion
import csv
from QST_nqubit import QStateTomo
def profile(fnc):
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

@profile
def code_to_profile(): 
    for _ in range(10): 
        data = DataSimNqubit(1,1000, tomo_type='MLE')
        expect_dict, expect = data.measure_qubit()
        del expect[0]
        qst = QStateTomo(expect,1)
        rho = qst.getRho()
        print(rho)

    ###################################
#%%
code_to_profile()