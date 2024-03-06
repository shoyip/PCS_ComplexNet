"""
PROBLEM 2: THE GIANT COMPONENT

In this problem we want to study the emergence of a giant component.

a) First of all, we generate instances of graphs using the configuration model
as defined in the previous problem. We use a Breadth First Search algorithm to
detect the largest connected component in each instance. For each instance we
can compute the largest connected component size, then average over them and
plot the behaviour of this value with respect to the value of the parameter pi.

b-c) We can generalize the analytical computation of the giant component size to
the case of the configuration model with a given degree distribution. It is
given by

TODO

"""

from compnet.graphs import ConfigModelDegreeGraph
import numpy as np
import matplotlib.pyplot as plt

# we pick equally spaced values of pi from 0.01 to 0.99
pi_values = np.linspace(0.01, 0.99, 15)

# we define the lists of means and standard deviations of giant component sizes
gcs_means, gcs_stds = [], []

# we iterate over values of the pi parameter
for pi in pi_values:

    # we define the number of vertices of a graph
    N = 1_000

    # we define the degree distribution
    degree_dict = {1: 1-pi, 4: pi}

    # we generate a ConfigModelDegreeGraph object instance
    G = ConfigModelDegreeGraph(N=N, degree_dict=degree_dict)

    # initialize the list of giant component sizes for random instances given pi
    gcs_samples = []

    # for each value of pi, we sample 20 instances
    for inst in range(20):

        # generate a graph instance and get the neighbourhoods
        G.generate_graph()
        G.find_neighbourhoods()

        # get the giant component size
        gcs_sample = G.get_giantcomponentsize() * 1. / N

        # append the value to the list of giant component sizes for a given pi
        gcs_samples.append(gcs_sample)

    # compute the mean of the giant component sizes
    gcs_means.append(np.mean(gcs_samples))

    # compute the standard deviation of the giant component sizes
    gcs_stds.append(np.std(gcs_samples) / np.sqrt(20))

fig = plt.figure()
plt.errorbar(pi_values, gcs_means, yerr=gcs_stds, fmt='.',
    label='GCS from random samples')
plt.grid()
plt.legend()
plt.title('Behaviour of Giant Component Size in the configuration model')
plt.xlabel(r'$\pi$')
plt.ylabel(r'Giant Component Size')
plt.close()

fig.savefig('assets/gcs.pdf', bbox_inches='tight')
