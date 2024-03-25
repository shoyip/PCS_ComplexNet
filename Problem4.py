"""
PROBLEM 4: THE FERROMAGNETIC ISING MODEL
"""

from compnet.ising import IsingConfigModelDegreeGraph, compute_autocorr
import matplotlib.pyplot as plt
import numpy as np

# HOW TO USE THE IsingConfigModelDegreeGraph CLASS

# let us define the parameters of the ising random graph
N = 200
pi = .3
degree_dict = {1: 1-pi, 4: pi}

# initialize the ising random graph object
G = IsingConfigModelDegreeGraph(N=N, degree_dict=degree_dict)

# instantiate a random graph
G.generate_graph()

# generate the dictionary of neighbourhoods
G.find_neighbourhoods()

# generate randomly a set of spins for the random graph
G.generate_spins()

# generate the dictionary of neighbouring spins for each node
G.find_spin_neighbourhoods()

def autocorr2(x):
    r2=np.fft.ifft(np.abs(np.fft.fft(x))**2).real
    c=(r2/x.shape-np.mean(x)**2)/np.std(x)**2
    return c[:len(x)//2]

def measure_autocorr():
    mcmc_sweeps = 2_000
    fig, axs = plt.subplots(4, 4, figsize=(30, 20))
    for i, T in enumerate([.5, 1.8, 2.6, 4.0]):
        for j, pi in enumerate([0.01, 0.11, 0.3, 0.7]):
            print(f'generating T={T}, pi={pi}')
            degree_dict = {1: 1-pi, 4: pi}
            G = IsingConfigModelDegreeGraph(N=N, degree_dict=degree_dict)
            G.generate_graph()
            G.find_neighbourhoods()
            G.generate_spins()
            G.find_spin_neighbourhoods()
            step_mags = np.abs(G.mcmc_wolff(T=T, mcmc_sweeps=mcmc_sweeps))
            acf = autocorr2(step_mags)
            axs[i][j].plot(acf)
            axs[i][j].set_title(f'$T={T}, \pi={pi}$')
    plt.close()
    
    fig.savefig('assets/wolff_mags.pdf', bbox_inches='tight')

if __name__ == "__main__":
    pass
