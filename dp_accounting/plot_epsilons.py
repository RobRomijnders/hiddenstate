import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm
import matplotlib

import pandas as pd
from prv_accountant.dpsgd import DPSGDAccountant
import gdp

sigma=15.0


cs = [5.4e-5,1e-4,2e-4]
epsilon_table_sc=[]

for c_ in cs:


    q = 1000/60000
    eps_error=0.01
    delta_error=1e-10
    max_epochs=400
    total_max_self_compositions=int(max_epochs/q)
    target_delta=1e-5

    # parameters for strongly convex setting
    # This is Theorem 4.5 of Bok et al. https://arxiv.org/pdf/2403.00278
    # We consider the "replace one" neighbouring relation of datasets,
    # and thus gradient sensitivity = 2
    mu = 2/sigma
    c = 1 - c_
    l = int(1/q)
    epsilons_sc = []
    for E_ in range(1,41):
    	E = 10*E_
    	mu_sc = np.sqrt(1 + c**(2*l - 2)* (1-c**2)/((1-c**l)**2) * (1 - c**(l*(E-1))) / (1 + c**(l*(E-1)))) * mu
    	ep_sc = gdp.gdp_to_ep_given_delta(mu_sc, target_delta)
    	epsilons_sc.append(ep_sc)
    epsilon_table_sc.append(epsilons_sc)


# this approximates the exact result Lemma 2.6 (Poisson subsampling for  "replace one" neighbouring relation of datasets)
# of https://arxiv.org/pdf/2407.04884
# very accurately
# The implementation of the exact result is also straightforward

accountant = DPSGDAccountant(
        noise_multiplier=sigma/2,
        sampling_probability=q,
        eps_error=eps_error,
        delta_error=delta_error,
        max_steps=total_max_self_compositions)


epsilons_dpsgd = []
epsilons_dpsgd2= []

epochs=[]


# DP-SGD accounting
for ii in range(40):
    ps_low, eps_estimate, eps_upper = accountant.compute_epsilon(num_steps=int((ii+1)*10/q),delta=target_delta)
    epsilons_dpsgd.append(eps_upper)
    epochs.append(int((ii+1)*10))

pp = PdfPages('./epsilons.pdf')

plot_ = plt.figure()
plt.rcParams.update({'font.size': 15.0})
plt.rc('text', usetex=True)
plt.rc('font', family='arial')
legs=[]
plt.xlabel('Epochs')
plt.ylabel(r'$\varepsilon$')

linestyles=['--',':','-.','-']

for (ii,epsilons_sc) in enumerate(epsilon_table_sc):
    plt.plot(epochs,epsilons_sc,linestyles[ii])


legs.append(r'NoisyCGD, $\eta \cdot \lambda = 5.4 \cdot 10^{-5}$')
legs.append(r'NoisyCGD, $\eta \cdot \lambda = 1 \cdot 10^{-4}$')
legs.append(r'NoisyCGD, $\eta \cdot \lambda = 2 \cdot 10^{-4}$')


plt.plot(epochs,epsilons_dpsgd,'-')
legs.append('DP-SGD')
plt.legend(legs,loc='lower right')
pp.savefig(plot_, bbox_inches = 'tight', pad_inches = 0)
pp.close()


plt.show()
plt.close()
