"""
PROBLEM 3: THE Q-CORE SIZE
"""

from compnet.graphs import ConfigModelDegreeGraph
from compnet.theoretical import get_theo_qcore_size, threecore_step
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import time

start = time.time()

pi_values = np.linspace(0.01, 0.99, 25)
N_values = [100, 500, 1000]

n_samples = 20

qcoresize_means, qcoresize_stds = [], []

qcs_meas = {N: {'means': [], 'stds': []} for N in N_values}

for N in N_values:
    qcs_values_means = []
    qcs_values_stds = []

    for pi in pi_values:
    
        print(f'Processing q-core values for pi={pi}')
    
        qcs_samples = []
    
        for _ in range(n_samples):
        
            degree_dict = {
                    1: 1-pi,
                    4: pi
            }
            
            G = ConfigModelDegreeGraph(N=N, degree_dict=degree_dict)
            
            G.generate_graph()
            
            G.find_neighbourhoods()
            
            qcs = G.get_biggest_qcoresize(q=3)
        
            qcs_samples.append(qcs / N)
    
        qcs_values_means.append(np.mean(qcs_samples))
        qcs_values_stds.append(np.std(qcs_samples) / np.sqrt(n_samples))

    qcs_meas[N]['means'] = qcs_values_means
    qcs_meas[N]['stds'] = qcs_values_stds

def get_threecoresize(pi):
    n_vertices = 10_000
    threecoresize, _ = get_theo_qcore_size(
        n_vertices = n_vertices,
        degree_distribution = [0, 1-pi, 0, 0, pi],
        q = 3,
        function=threecore_step
    )
    return threecoresize / n_vertices

## generate example ode figure
#
#pi = .3
#_, U = get_theo_qcore_size(
#        n_vertices = 1_000,
#        degree_distribution = [0, 1-pi, 0, 0, pi],
#        q = 3,
#        function = threecore_step)
#
#fig = plt.figure()
#for i in range(5):
#    plt.plot(U[:-1, i], label=f'$d={i}$')
#plt.title('Nodes per degree in time for stochastic 3-core for $pi=0.3$')
#plt.xlabel(r'$T$')
#plt.ylabel(r'$p_d(t)$')
#plt.grid()
#plt.legend()
#plt.close()
#
#fig.savefig('assets/ode_evol_pi03.pdf', bbox_inches='tight')
#
#pi = .8
#_, U = get_theo_qcore_size(
#        n_vertices = 1_000,
#        degree_distribution = [0, 1-pi, 0, 0, pi],
#        q = 3,
#        function = threecore_step)
#
#fig = plt.figure()
#for i in range(5):
#    plt.plot(U[:-1, i], label=f'$d={i}$')
#plt.title('Nodes per degree in time for stochastic 3-core for $pi=0.8$')
#plt.xlabel(r'$T$')
#plt.ylabel(r'$p_d(t)$')
#plt.grid()
#plt.legend()
#plt.close()
#
#fig.savefig('assets/ode_evol_pi08.pdf', bbox_inches='tight')

# computing theoretical values

print("Computing theoretical q-core sizes...")

get_threecoresize_vect = np.vectorize(get_threecoresize)

theo_pi_values = np.linspace(0.1, 0.99, 50)

theo_qcoresizes = get_threecoresize_vect(theo_pi_values)

fig = plt.figure()
for N in N_values:
    plt.errorbar(x=pi_values, y=qcs_meas[N]['means'],
                 yerr=qcs_meas[N]['stds'], fmt='.',
                 label=f'3-core sizes for $N={N}$')
plt.plot(theo_pi_values,
         theo_qcoresizes,
         label='Theoretical $q$-core sizes')
plt.xlabel(r'$\pi$')
plt.ylabel(r'Relative size of 3-core')
plt.title(r'Behaviour of 3-core size in the configuration model')
plt.grid()
plt.legend()
plt.close()

fig.savefig('assets/qcore.pdf', bbox_inches='tight')

stop = time.time()

print(f'Time elapsed: {(stop-start)/60:.0f}m{(stop-start)%60:.0f}s')
