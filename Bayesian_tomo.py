#%%
from matplotlib import transforms
import numpy as np
from qutip import *
import Bayesian_tools as btool
from Data_sim import DataSimNqubit
import matplotlib.pyplot as plt
import matplotlib.patches as patches
class BayInfTomo():
    """Class that implements Bayesian inference Tomography
    """
    def __init__(self, n_count, povm, nqubits, samples=10000, burn_in_per=0.5, deviation=0.05, rho_guess=None, tol=0.01):
        self.n_count = n_count
        self.povm = povm
        self.nqubits = nqubits
        self.dim = 2**nqubits
        self.samples = int(samples*(1-burn_in_per))
        self.burn_in = int(samples*burn_in_per)
        self.rho_guess = rho_guess
        self.deviation = deviation
        self.tol_vol = tol*((self.dim**2)-1)
        self.accept_count = 0

    def burn_in_phase(self):
        theta_distrib = np.zeros(shape=(self.burn_in,(self.dim**2)-1))
        thetha = btool.calculate_prior_thetha(self.dim, self.rho_guess)
        for i in range(self.burn_in):
            thetha_cand = btool.sample_thetha_cand(self.dim, thetha, self.deviation, self.edge_set)
            joint_posterior = btool.calculate_joint_posterior_ratio(thetha, thetha_cand, self.dim, self.povm, self.n_count)
            proposal_ratio = btool.calculate_ratio_proposal(thetha, thetha_cand, self.dim, self.deviation, self.edge_set)
            thetha, self.accept_count = btool.accept_crit(thetha, thetha_cand, joint_posterior, proposal_ratio, self.accept_count)
            theta_distrib[i,:] = thetha

            if (i+1) % (self.burn_in/5) == 0:
                accept_rate = self.accept_count/(i+1)
                print(f"Acceptance rate is {accept_rate*100}% after {(i+1)*100/self.burn_in}% of the burn in.")
                self.deviation = btool.update_dev(self.deviation, accept_rate)
                print(f"Deviation is updated to {self.deviation}.")
                self.accept_count = 0
        self.accept_count = 0
        return thetha, theta_distrib

    def sampling_phase(self, thetha):
        self.edge_set = 0
        if isinstance(self.edge_set, int) == False:
            thetha = btool.post_bin_prior_theta(self.dim,self.edge_set)
        theta_distrib = np.zeros(shape=(self.samples,(self.dim**2)-1))
        for i in range(self.samples):
            thetha_cand = btool.sample_thetha_cand(self.dim, thetha, self.deviation, self.edge_set)
            joint_posterior = btool.calculate_joint_posterior_ratio(thetha, thetha_cand, self.dim, self.povm, self.n_count)
            proposal_ratio = btool.calculate_ratio_proposal(thetha, thetha_cand, self.dim, self.deviation, self.edge_set)
            thetha, self.accept_count = btool.accept_crit(thetha, thetha_cand, joint_posterior, proposal_ratio, self.accept_count)
            theta_distrib[i,:] = thetha
        print(f"Acceptance rate is {(self.accept_count/self.samples)*100}% in the sampling phase.")
        self.accept_count = 0
        return theta_distrib

    def getRho(self):
        self.edge_set = 0
        # Burn in phase
        thetha, theta_distrib_burn = self.burn_in_phase()
        # Sampling phase
        theta_distrib_samp = self.sampling_phase(thetha)
        self.edge_set = btool.bin_vectors(theta_distrib_samp)
        rho = Qobj(np.mean(np.array([Qobj(btool.op_cholesky(btool.construct_t(list(i), self.dim))) for i in theta_distrib_samp]), axis=0), dims=[[2]*self.nqubits,[2]*self.nqubits])
        return rho, theta_distrib_burn, theta_distrib_samp
#%%
def plot_thetas(rho, thetha_dis_burn, thetha_dis_samp, nqubits):
    thet = btool.get_true_theta(rho, nqubits)
    thetha_dis = np.zeros(shape=(len(thetha_dis_burn)+len(thetha_dis_samp),len(thetha_dis_burn[0])))
    thetha_dis[:len(thetha_dis_burn),:] = thetha_dis_burn
    thetha_dis[len(thetha_dis_burn):len(thetha_dis_burn)+len(thetha_dis_samp),:] = thetha_dis_samp
    for j,i in enumerate(thet):
        fig_i, ax_i = plt.subplots(figsize=(16,10))
        ax_i.plot(np.linspace(1,len(thetha_dis[:,j]),len(thetha_dis[:,j])),thetha_dis[:,j])
        ax_i.hlines(i, 0, len(thetha_dis[:,j]),color='r')
        rect = patches.Rectangle((0,min(thetha_dis_burn[:,j])), len(thetha_dis_burn[:,j]), max(thetha_dis_burn[:,j])-min(thetha_dis_burn[:,j]),color='black',alpha=0.3,transform=ax_i.transData)
        ax_i.add_patch(rect)
        ax_i.set_ylim(min(thetha_dis[:,j])+0.05,max(thetha_dis[:,j])+0.05)
        ax_i.set_xlim(0,len(thetha_dis))
        ax_i.set_yticks([0, np.pi/4, 2*np.pi/4, 3*np.pi/4,np.pi])

        ax_i.set_yticklabels(['0', '$\pi/4$','$2\pi/4$','$3\pi4$', '$\pi$'])
        fig_i.tight_layout()