"""
PROBLEM 4: THE FERROMAGNETIC ISING MODEL
"""

from compnet.ising import IsingConfigModelDegreeGraph, autocorr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from joblib import Parallel, delayed
import numpy as np
import time

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
            sweep_mags = np.abs(G.mcmc_wolff(T=T, mcmc_sweeps=mcmc_sweeps))
            acf = autocorr(sweep_mags)
            axs[i][j].plot(acf)
            axs[i][j].set_title(f'$T={T}, \pi={pi}$')
    plt.close()
    
    fig.savefig('assets/wolff_mags.pdf', bbox_inches='tight')

#def make_mcmc_heatmap():
#    mcmc_sweeps = 500
#    eq_sweeps = 100
#    sample_sweeps = 20
#
#    Ts = np.linspace(0.01, 4., 7)
#    pis = np.linspace(0.01, 1., 15)
#
#    heatmap_matrix_means = np.zeros((7, 15))
#    heatmap_matrix_stds = np.zeros((7, 15))
#
#    for i in range(7):
#        for j in range(15):
#            start = time.time()
#            T = Ts[i]
#            pi = pis[j]
#            print(f'Computing mag value for T={T}, pi={pi}...')
#            sweep_mags = []
#            for iteration in range(1):
#                degree_dict = {1: 1-pi, 4: pi}
#                G = IsingConfigModelDegreeGraph(N=N, degree_dict=degree_dict)
#                G.generate_graph()
#                G.find_neighbourhoods()
#                G.generate_spins()
#                G.find_spin_neighbourhoods()
#                sweep_mag = np.abs(
#                        G.mcmc_wolff(
#                            T=T,
#                            mcmc_sweeps=mcmc_sweeps,
#                            return_all=False,
#                            eq_sweeps=eq_sweeps,
#                            sample_sweeps=sample_sweeps))
#                sweep_mags.append(sweep_mag)
#            heatmap_matrix_means[i][j] = np.mean(sweep_mags)
#            heatmap_matrix_stds[i][j] = np.std(sweep_mags) / np.sqrt(20)
#            stop = time.time()
#            print(f'Took {(stop-start)//60:.0f}m{(stop-start)%60:.0f}s')
#
#    fig, ax = plt.subplots()
#    im = plt.imshow(heatmap_matrix_means, cmap='hot')
#    plt.xticks(Ts)
#    plt.yticks(pis)
#    plt.xlabel(r'$T$')
#    plt.ylabel(r'$\pi$')
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    plt.colorbar(im, cax=cax)
#    plt.title('Heatmap for MCMC magnetization values')
#    plt.close()
#    fig.savefig('assets/mcmc_heatmap.pdf', bbox_inches='tight')

def make_mcmc_samples():
    nT = 10
    npi = 20

    M = 1
    N = 100

    Ts = np.linspace(0.01, 4., nT)
    pis = np.linspace(0.01, 1., npi)

    mcmc_sweeps = 500
    eq_sweeps = 100
    sample_sweeps = 20

    mcmc_samples = np.zeros((nT, npi, M))

    def mcmc_iteration(N, pi, T, mcmc_sweeps, eq_sweeps, sample_sweeps):
        degree_dict = {1: 1-pi, 4: pi}
        G = IsingConfigModelDegreeGraph(N=N, degree_dict=degree_dict)
        G.generate_graph()
        G.find_neighbourhoods()
        G.generate_spins()
        G.find_spin_neighbourhoods()
        equilibration_mags = G.mcmc_wolff(T)
        instance_mag = np.mean(np.abs(equilibration_mags))
        print(f'== TEMP {T:.2f} PI {pi:.2f} MAG {instance_mag:.2f}')
        return instance_mag

    for i, T in enumerate(Ts):
        for j, pi in enumerate(pis):
            start = time.time()
            #print(f'Computing mag value for T={T}, pi={pi}...')
            instances_mags = Parallel(n_jobs=4)(delayed(mcmc_iteration)(N, pi, T, mcmc_sweeps, eq_sweeps, sample_sweeps) for m in range(M))
            mcmc_samples[i, j] = np.array(instances_mags)
            print(f'{i*nT+j}/{nT*npi} TEMP {T:.2f} PI {pi:.2f} MAG {np.mean(np.abs(instances_mags)):.2f}')
            stop = time.time()
            print(f'Took {(stop-start)//60:.0f}m{(stop-start)%60:.0f}s')

    np.save('data/prova_mcmc_mags.npy', mcmc_samples)

def make_bp_samples():
    nT = 10
    npi = 20

    M = 50
    N = 100

    Ts = np.linspace(0.01, 4., nT)
    pis = np.linspace(0.01, 1., npi)

    bp_samples = np.zeros((nT, npi, M))

    def bp_iteration(T, pi):
        beta = 1. / T
        degree_dict = {1: 1-pi, 4: pi}
        G = IsingConfigModelDegreeGraph(N=N, degree_dict=degree_dict)
        G.generate_graph()
        G.find_bp_cavity_fields(beta=beta, max_iter=100)
        G.find_bp_fields(beta=beta)
        mag = (np.mean(np.tanh(beta * G.fields)))
        return mag

    for i, T in enumerate(Ts):
        for j, pi in enumerate(pis):
            start = time.time()
            #print(f'Computing mag value for T={T}, pi={pi}...')
            instances_mags = Parallel(n_jobs=4)(delayed(bp_iteration)(T, pi) for m in range(M))
            bp_samples[i, j] = np.array(instances_mags)
            print(f'{i*nT+j}/{nT*npi} TEMP {T:.2f} PI {pi:.2f} MAG {np.mean(np.abs(instances_mags)):.2f}')
            stop = time.time()
            print(f'Took {(stop-start)//60:.0f}m{(stop-start)%60:.0f}s')

    np.save('data/bp_mags.npy', bp_samples)

#    fig, ax = plt.subplots()
#    plt.pcolormesh(pis, Ts, np.abs(heatmap_matrix_means), shading='nearest')
##    plt.imshow(heatmap_matrix_means, origin='lower')
#    plt.xlabel(r'$\pi$')
#    plt.ylabel(r'$T$')
##    plt.xlim([0, 1])
##    plt.ylim([0, 4])
#    plt.colorbar(label=r'$|\vec m|$')
#    plt.title('Heatmap for BP magnetization values')
#    plt.close()
#    fig.savefig('assets/bp_heatmap.pdf', bbox_inches='tight')

if __name__ == "__main__":
    # measure_autocorr()
    make_mcmc_samples()
    #make_bp_samples()
