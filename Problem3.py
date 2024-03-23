"""
PROBLEM 3: THE Q-CORE SIZE
"""

from compnet.graphs import ConfigModelDegreeGraph
#from compnet.theoretical
import numpy as np
import matplotlib.pyplot as plt

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
    
        qcoresizes.append(qcoresize)

    qcoresize_means.append(np.mean(qcoresizes))
    qcoresize_stds.append(np.std(qcoresizes) / np.sqrt(n_samples))

fig = plt.figure()
plt.errorbar(x=pi_values, y=qcoresize_means,
             yerr=qcoresize_stds, fmt='.',
             label='$q$-core sizes on RG instances')
plt.legend()
plt.close()

fig.savefig('assets/qcore.pdf', bbox_inches='tight')
