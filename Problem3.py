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

pi_values = np.linspace(0.01, 0.99, 15)

N = 1000

n_samples = 20

qcoresize_means, qcoresize_stds = [], []

for pi in pi_values:

    print(f'Processing q-core values for pi={pi}')

    qcoresizes = []

    for sample in range(n_samples):
    
        degree_dict = {
                1: 1-pi,
                4: pi
        }
        
        G = ConfigModelDegreeGraph(N=N, degree_dict=degree_dict)
        
        G.generate_graph()
        
        G.find_neighbourhoods()
        
        qcoresize = G.get_biggest_qcoresize(q=3)
    
        qcoresizes.append(qcoresize / N)

    qcoresize_means.append(np.mean(qcoresizes))
    qcoresize_stds.append(np.std(qcoresizes) / np.sqrt(n_samples))

def get_threecoresize(pi):
    threecoresize, _ = get_theo_qcore_size(
        n_vertices = 10_000,
        degree_distribution = [0, 1-pi, 0, 0, pi],
        q = 3,
        function=threecore_step
    )
    return threecoresize / 10_000

# computing theoretical values

print("Computing theoretical q-core sizes...")

get_threecoresize_vect = np.vectorize(get_threecoresize)

# theo_qcoresize, _ = get_theo_qcore_size(
#         n_vertices = 10_000,
#         degree_distribution=[0, 1-pi, 0, 0, pi],
#         q=3,
#         function=threecore_step)

theo_pi_values = np.linspace(0.1, 0.99, 40)

theo_qcoresizes = get_threecoresize_vect(theo_pi_values)

fig = plt.figure()
plt.errorbar(x=pi_values, y=qcoresize_means,
             yerr=qcoresize_stds, fmt='.',
             label='$q$-core sizes for $N=1000$')
plt.plot(theo_pi_values,
         theo_qcoresizes,
         label='Theoretical $q$-core sizes')
plt.grid()
plt.legend()
plt.close()

fig.savefig('assets/qcore.pdf', bbox_inches='tight')

stop = time.time()

print(f'Time elapsed: {(stop-start) / 60} mins')
