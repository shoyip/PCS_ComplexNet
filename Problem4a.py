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

def make_mcmc_samples():
    nT = 10
    npi = 20

    M = 50
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

    np.save('data/mcmc_mags.npy', mcmc_samples)

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

def plot_heatmap(sample_file, outfile, name):
    nT = 10
    npi = 20
    mag_samples = np.load(sample_file)
    Ts = np.linspace(0.01, 4., nT)
    pis = np.linspace(0.01, 1., npi)
    mag_means = np.mean(np.abs(mag_samples), axis=2)
    mag_stds = np.std(np.abs(mag_samples), axis=2)
    fig, axs = plt.subplots(2, 1, figsize=(15, 12))
    im = axs[0].pcolormesh(pis, Ts, mag_means, shading='nearest')
    axs[1].pcolormesh(pis, Ts, mag_stds, shading='nearest')
    plot_phasediag(axs[0])
    plot_phasediag(axs[1])
    axs[0].title.set_text(f'{name} means')
    axs[1].title.set_text(f'{name} stds')
    axs[0].set_xlabel(r'$\pi$')
    axs[0].set_ylabel(r'$T$')
    axs[1].set_xlabel(r'$\pi$')
    axs[1].set_ylabel(r'$T$')
    axs[0].set_xlim([0, 1])
    axs[0].set_ylim([0, 4])
    axs[1].set_xlim([0, 1])
    axs[1].set_ylim([0, 4])
    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, label=r'$|\vec m|$', cax=cax)
#    plt.title(f'Heatmap for {name}')
#    plt.show()
    plt.close()
    fig.savefig(outfile, bbox_inches='tight')

def plot_pd_heatmap(sample_file, outfile, name):
    nT = 9
    npi = 20
    mag_samples = np.load(sample_file)
    Ts = np.linspace(0.01, 4., nT)
    pis = np.linspace(0.01, 1., npi)
    mags = mag_samples
    fig, axs = plt.subplots()
    im = axs.pcolormesh(pis, Ts, mags[1:, :], shading='nearest')
    axs.set_xlabel(r'$\pi$')
    axs.set_ylabel(r'$T$')
    axs.set_xlim([0, 1])
    axs.set_ylim([0, 4])
    plot_phasediag(axs)
    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, label=r'$|\vec m|$', cax=cax)
#    plt.title(f'Heatmap for {name}')
#    plt.show()
    plt.close()
    fig.savefig(outfile, bbox_inches='tight')

def plot_phasediag(ax):
    pis_fx = np.linspace(1./9., 1, 1_000)
    Ts_fx = 1 / np.arctanh((1+3*pis_fx)/(12*pis_fx))
    ax.plot(pis_fx, Ts_fx, label='Theoretical curve', color='white')

if __name__ == "__main__":
    # measure_autocorr()
    # make_mcmc_samples()
    # make_bp_samples()
    plot_heatmap('data/mcmc_mags.npy', 'assets/mcmc_heatmap.pdf', 'MCMC')
    plot_heatmap('data/bp_mags.npy', 'assets/bp_heatmap.pdf', 'BP')
    plot_pd_heatmap('data/pd_mags.npy', 'assets/pd_heatmap.pdf', 'PD')
