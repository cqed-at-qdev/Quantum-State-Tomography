#%%
import numpy as np
import qutip
from qutip import *
from scipy.optimize import minimize
import settings
#---------------------------------------------------------------------#
#-------------------------------Functions-----------------------------#
#---------------------------------------------------------------------# 
def MLE_QST(measurments, Paulis, n, datafile):
    """Takes the measured values, Paulis and number of qbits to calculate the most likely t.
       Here t is the cholesky decomposed matrix of the density matrix.

    Args:
        measurments ([list]): Takes a list of the three expectation values of the state.
        Paulis ([list]): Takes the three pauli matrices as input.
        n ([integer]): Number of qubits to do tomography on.
        datafile ([list]): Expectation values of state.
    Returns:
        [array]: Outputs the cholesky decomposed density matrix.
    """
    # Initial guess for the MLE.
    t_tunable = np.ones(int(4**n))/4**n
    # Takes a initial frame for animation.
    rho_check = op_cholesky(t_tunable)
    rho_check = Qobj(rho_check)
    settings.rho_anim_list.append(rho_check)
    # Calculates a inital trace distance between start guess and true rho.
    settings.rho_basic = (0.5 * (qeye(2)+datafile[0]*sigmax()+datafile[1]*sigmay()+datafile[2]*sigmaz()))
    d = tracedist(settings.rho_basic, rho_check)
    settings.tracedist_list.append(d)

    consts = ({'type': 'eq',
               'method': 'BFGS',
               'fun': lambda t_tunable: Trace1constraint(t_tunable)})
    result = minimize(MLE_Function_QST,
                        t_tunable, 
                        args=(measurments,Paulis),
                        constraints=consts,
                        method='SLSQP', 
                        tol=1e-15,callback = callbackF, options={'disp': True})
    result = result['x']
    print(result)
    return result

def callbackF(t_tunable):
    """Callback function will be called between each iteration of the MLE, it turns the current t into a density matrix with the cholesky decomp func and saves it to the animation list.
       It also calculates the trace distance between the current guess and the true state.

    Args:
        t_tunable ([array]): The current MLE guess.
    """
    rho_MLE = op_cholesky(t_tunable)
    rho_MLE = Qobj(rho_MLE)
    settings.rho_anim_list.append(rho_MLE)
    d = tracedist(settings.rho_basic, rho_MLE)
    settings.tracedist_list.append(d)
    settings.Nfeval += 1

# Keeps the trace of t physical
def Trace1constraint(t_tunable):
    """Keeps the matrix physical and within the Bloch sphere.

    Args:
        t_tunable ([array]): The current MLE guess.

    Returns:
        [array]: The current MLE guess, normalized.
    """
    return np.array((t_tunable[:]**2).sum()-1)
 
def MLE_Function_QST(t_tunable, measurments, Paulis):
    """The function we minimize when the MLE function is run. 
       It calculates rho using cholensky decomposition using an inital guess for T.

    Args:
        t_tunable ([array]): The current MLE guess.
        measurments ([list]): Takes a list of the three expectation values of the state.
        Paulis ([list]): Takes the three pauli matrices as input.

    Returns:
        [float]: MLE guess.
    """
    rho = op_cholesky(t_tunable)
    expect = np.einsum('ij,ljk',rho,Paulis)
    tr_expect = np.einsum('iij', expect).real
    L = np.sum((measurments-tr_expect)**2)
    return L

def op_cholesky(t_tunable):
    """Function uses Cholesky decomposition to take the matrix t and turn it into the density matrix.

    Args:
        t_tunable ([array]): The current MLE guess.

    Returns:
        [array]: Density matrix constructed from t.
    """
    n = int(np.log2(np.sqrt(len(t_tunable))))
    d = 2
    T = np.zeros((d**n, d**n)) + 0j
    main_diag = np.diag(t_tunable[0:int(d**n)])

    even = t_tunable[int(d**n)::2]
    odd = t_tunable[int(d**n+1)::2]

    off = even + 1j*odd
    T += main_diag

    for i in range(1, int(d**n)):
        diag = np.diag(off[int((i-1)*d**n-(i-1)*i//2):int((i)*d**n-i*(i+1)//2)], k=-i)
        T += diag
    T = np.matrix(T)
    norm = np.array(T.H.dot(T).trace()).flatten()[0].real
    op_t = (T.H.dot(T))/norm
    op_t = np.array(op_t)
    return op_t