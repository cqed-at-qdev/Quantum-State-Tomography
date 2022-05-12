import numpy as np
from qutip import *
from scipy.optimize import minimize
import settings
#---------------------------------------------------------------------#
#-------------------------------Functions-----------------------------#
#---------------------------------------------------------------------# 
def MLE_QST(measurments, Paulis, n):
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
    rho_check = op_cholesky(t_tunable)
    rho_check = Qobj(rho_check,dims=[[2]*n,[2]*n])
    settings.rho_MLE.append(rho_check)
    # Calculates a initial trace distance between start guess and true rho.
    #d = tracedist(settings.rho_true, rho_check)
    #settings.tracedist_list.append(d)

    consts = ({'type': 'eq',
               'method': 'BFGS',
               'fun': lambda t_tunable: Trace1constraint(t_tunable)})
    result = minimize(MLE_Function_QST,
                        t_tunable, 
                        args=(measurments,Paulis),
                        constraints=consts,
                        method='SLSQP', 
                        tol=1e-5, callback=callbackF,options={'disp': False})
    result = result['x']
    return result

def callbackF(t_tunable):
    """Callback function will be called between each iteration of the MLE, it turns the current t into a density matrix with the cholesky decomp func and saves it to the animation list.
       It also calculates the trace distance between the current guess and the true state.

    Args:
        t_tunable ([array]): The current MLE guess.
    """
    rho_MLE = op_cholesky(t_tunable)
    rho_MLE = Qobj(rho_MLE,dims=[[2]*(int(np.log2(len(t_tunable))/2)),[2]*(int(np.log2(len(t_tunable))/2))])
    """d = tracedist(settings.rho_true, rho_MLE)
    print(d)
    settings.tracedist_list.append(d)
    settings.Nfeval += 1"""
    settings.rho_MLE.append(rho_MLE)

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
        measurments ([list]): List of expectation values for the state.
        Paulis ([list]): List of the POVM (Pauli matrices in this case).

    Returns:
        L ([float]): Likelihood value
    """
    rho = op_cholesky(t_tunable)
    expect = np.einsum('ij,ljk',rho,Paulis)  # perform matrix multiplication over array of paulis
    tr_expect = np.einsum('iij', expect).real # take the real component of the trace

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

def get_rho_true(a,b):
    basis0 = (basis(2,0))/(basis(2,0)).norm()
    basis1 = (basis(2,1))/(basis(2,1)).norm()
    psi = (a*basis0+b*basis1)/(a*basis0+b*basis1).norm()
    psi_com_con = psi.conj().trans()
    rho_true = psi*psi_com_con
    return rho_true