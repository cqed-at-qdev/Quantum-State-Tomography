#%%
from cmath import nan
from re import S, X
import numpy as np
from scipy.stats import truncnorm, norm
from qutip import *
def sample_thetha_cand(dim, thetha, deviation, edge_set):
    """Samples the new theta vector from i truncated normal distributions.

    Args:
        dim ([int]): number of dimensions in the quantum system.
        thetha ([list]): list of the old theta vector.
        deviation ([ffloat]): standard deviation of the distributions.

    Returns:
        thetha_cand ([list]): a list of the new proposed theta vector.
    """
    if isinstance(edge_set, int) == False:
        thetha_cand = []
        for i in range(len(thetha)):
            # Select the interval to truncate the normal distribution in.
            a, b = (edge_set[i,0]-thetha[i])/deviation, (edge_set[i,1]-thetha[i])/deviation
            # Sample new theta
            thetha_cand_i = truncnorm.rvs(a,b, loc=thetha[i], scale=deviation)
            thetha_cand.append(thetha_cand_i)
    else:
        thetha_cand = []
        for i in range(len(thetha)):
            # Select the interval to truncate the normal distribution in.
            if 1 <= (i+1) <= dim-1:
                a,b = (0-thetha[i])/deviation, (np.pi/2-thetha[i])/deviation
            elif dim <= (i+1) <= (dim**2)-1:
                a,b = (0-thetha[i])/deviation, (np.pi-thetha[i])/deviation
            # Sample new theta
            thetha_cand_i = truncnorm.rvs(a,b, loc=thetha[i], scale=deviation)
            thetha_cand.append(thetha_cand_i)
    return thetha_cand

def accept_crit(thetha, thetha_cand, joint_posterior, proposal_ratio, accept_count):
    """Function that determines whether the new theta vector is accepted.

    Args:
        thetha ([list]): old theta vector.
        thetha_cand ([list]): new theta vector
        joint_posterior ([float]): the joint posterior ratio
        proposal_ratio ([float]): the joint proposal ratio

    Returns:
        thetha ([list]): either the old or new theta vector depending on whether it was accepted.
    """
    # Sum the two log functions
    post_prop = joint_posterior+proposal_ratio
    # Cancel out the log
    post_prop = np.exp(post_prop)

    # Choose the minimum between 1 and the post_prop
    min_array = np.array([1, post_prop])
    min_val = np.nanmin(min_array)
    # If 1 accept with 100% prob
    if min_val == 1:
        thetha = thetha_cand
        accept_count +=1
    # Else accept with prob of post_prob
    else:
        choice = np.random.choice([True, False], p=[min_val, 1-min_val])
        if choice == True:
            thetha = thetha_cand
            accept_count +=1  
        elif choice == False:
            pass
        else:
            print('Error something is wrong')
            exit()
    return thetha, accept_count

def calculate_prior_thetha(dim, rho_guess):
    """Calculates the prior thetha

    Args:
        dim ([int]): number of dimensions in the quantum system
        rho_guess ([Qobj]): best guess of the output state if one has a guess.

    Returns:
        _type_: _description_
    """
    thetha = []
    # Generate uniform prior thetha
    if rho_guess == None:
        for i in range((dim**2)-1):
            if 1 <= (i+1) <= dim-1:
                a,b = 0, np.pi/2
            elif dim <= (i+1) <= (dim**2)-1:
                a,b = 0, np.pi
            thetha_i = np.random.uniform(a,b)
            thetha.append(thetha_i)
    # Generate prior thetha from estimate of rho
    else:
        t_array = np.linalg.cholesky(rho_guess)
        t_diag = [i for i in np.diag(t_array)]
        t_diag = t_diag[1:]
        t_column = [i.real for i in t_array[1:,0]]
        t_final = t_array[-1,0].imag
        t_list = [*t_diag, *t_column, t_final]
        t_list = np.flip(t_list)
        for i in range(len(t_list)):
            if i==0:
                thetha_i = np.arccos(t_list[i])
                thetha.append(thetha_i)
            else:
                sin_prod = np.prod([np.sin(x) for x in thetha])
                thetha_i = np.arccos(t_list[i]/sin_prod)
                thetha.append(thetha_i)
        thetha = np.flip(thetha)
    return thetha

def post_bin_prior_theta(dim, edge_set):
    thetha = []
    # Generate uniform prior thetha
    for i in range((dim**2)-1):
        a, b = edge_set[i,0], edge_set[i,1]
        thetha_i = np.random.uniform(a,b)
        thetha.append(thetha_i)
    return thetha

def uniform_prior(thetha, dim):
    prior_distrib = []
    for i in range((dim**2)-1):
        if 1 <= (i+1) <= dim-1:
            a,b = 0, np.pi/2
        elif dim <= (i+1) <= (dim**2)-1:
            a,b = 0, np.pi
        mean = (a+b)/2
        dev = np.sqrt((1/12)*((b-a)**2))
        prio = np.exp(-0.5*(((thetha[i]-mean)/dev)**2))
        prior_distrib.append(prio)
    prior_distrib = np.prod(prior_distrib)
    return prior_distrib

def calculate_joint_posterior_ratio(thetha, thetha_cand, dim, povm, n_count):
    """Calculates the joint posterior ratio.

    Args:
        thetha ([list]): old theta vector
        thetha_cand ([list]): new theta vector
        dim ([int]): dimensions in the quantum system
        povm ([list]): list of all povm's used in the measurment of the system
        n_count ([list]): number of counts for each outcome/povm

    Returns:
        joint_posterior ([float]): the ratio of the joint posterior of the old vs the new theta
    """
    # Create copies of both theta and theta_cand
    thetha_copy = thetha.copy()
    thetha_cand_copy = thetha_cand.copy()
    # Construct the corresponding T matrices 
    t_old = construct_t(thetha_copy, dim)
    t_new = construct_t(thetha_cand_copy, dim)
    # In turn construct the corresponding density matrices
    rho_old = Qobj(op_cholesky(t_old))
    rho_new = Qobj(op_cholesky(t_new))
    # Calculate the probabilities of the density matrix given all POVM
    p_old = [(rho_old*j).tr().real for j in povm]
    p_new = [(rho_new*j).tr().real for j in povm]
    #prior_distrib_old = uniform_prior(thetha, dim)
    #prior_distrib_new = uniform_prior(thetha_cand, dim)

    # Calculate the logaritm of the joint posterior ratio.
    #joint_posterior = np.prod([(p_new[j]/p_old[j])**n_count[j] for j in range(len(n_count))])
    joint_posterior = np.sum([n_count[j]*np.log(np.float64(p_new[j])/p_old[j]) for j in range(len(n_count))])

    return joint_posterior

def calculate_ratio_proposal(thetha, thetha_cand, dim, deviation, edge_set):
    """Calculate the proposal ratio.

    Args:
        thetha ([list]): old theta vector
        thetha_cand ([list]): new theta vector
        dim ([int]): dimensions in the quantum state
        deviation ([float]): deviation of the distribution.

    Returns:
        proposal_ratio ([float]): the proposal_ratio
    """
    ratio_list = []
    for i in range(len(thetha)):
        if isinstance(edge_set, int) == False:
            a, b = edge_set[i,0], edge_set[i,1]
        else:
            # Choose the limits of the distributions
            if 1 <= (i+1) <= dim-1:
                a,b = 0, np.pi/2
            elif dim <= (i+1) <= (dim**2)-1:
                a,b = 0, np.pi
        # Calculate the four cumulative density functions

        old_cdf_a = norm.cdf((a-thetha[i])/deviation, loc=thetha[i], scale=deviation)
        old_cdf_b = norm.cdf((b-thetha[i])/deviation, loc=thetha[i], scale=deviation)
        new_cdf_a = norm.cdf((a-thetha_cand[i])/deviation, loc=thetha_cand[i], scale=deviation)
        new_cdf_b = norm.cdf((b-thetha_cand[i])/deviation, loc=thetha_cand[i], scale=deviation)
        # Calculate the log of the ratio
        #ratio = (old_cdf_b-old_cdf_a)/(new_cdf_b-new_cdf_a)
        ratio = np.log((old_cdf_b-old_cdf_a)/(new_cdf_b-new_cdf_a))
        ratio_list.append(ratio)
    # Sum all the terms
    #proposal_ratio = np.prod(ratio_list)
    proposal_ratio = np.sum(ratio_list)
    return proposal_ratio

def construct_t(thetha_inst, dim):
    """Construct array of t from theta vector

    Args:
        thetha_inst ([list]): Instance of the current theta vector
        dim ([int]): dimensions of the quantum state

    Returns:
        t_array ([array]): An array of the reconstructed t's
    """
    t_array = np.zeros(dim**2)
    for i in range(len(t_array)):
        if i == 0:
            thetha_list = np.flip(thetha_inst)
            product_list = [np.sin(x) for x in thetha_list]
            t_array[i] = np.prod(product_list)
        else:
            thetha_list = np.flip(thetha_inst)
            product_list = [np.sin(x) for x in thetha_list]
            product_list[-1] = np.cos(thetha_list[-1])
            t_array[i] = np.prod(product_list)
            del thetha_inst[0]
    return t_array

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

def bin_vectors(theta_distrib):
    binned_theta, edges = np.histogramdd(theta_distrib, bins=int(np.sqrt(len(theta_distrib))))
    max_count_idx = np.unravel_index(binned_theta.argmax(), binned_theta.shape)
    edge_set = np.zeros(shape=(len(theta_distrib[1]),2))
    for i in range(len(edge_set)):
        k = max_count_idx[i]
        edge_set[i,:] = [edges[i][k],edges[i][k+1]]
    
    return edge_set

def tolerance_check(edge_set, tol_vol):
    diff = np.absolute(np.diff(edge_set))
    hyper_vol = np.prod(diff)
    if hyper_vol <= tol_vol:
        cont = False
    else:
        cont = True
    
    return cont

def bin_with_max_count(theta_distrib, edge_set):
    theta_list = []
    for j in theta_distrib:
        x = [True for i,k in enumerate(j) if edge_set[i,0] <= k <= edge_set[i,1]]
        if len(x) == len(j):
            theta_list.append(j)
        else:
            pass
    theta_guess = list(np.mean(np.array(theta_list), axis=0))

    return theta_guess

def update_dev(deviation, accept_rate):
    accept_opt = 0.3
    accept_diff = accept_rate-accept_opt
    x = 1 + accept_diff
    new_deviation = x * deviation
    return new_deviation

def get_true_theta(rho, n_qubits):
    eig = rho.eigenenergies()
    print(eig)
    while all(i > 0 for i in eig)==False:
        rho = rho + 1e-10 * tensor([qeye(2)]*n_qubits)
        eig = rho.eigenenergies()
        print(eig)
    thetha_true = calculate_prior_thetha(2*n_qubits,rho).real
    return thetha_true