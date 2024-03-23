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

   gamma = 1 - (1-pi) * (1-sqrt(pi)) / (2*sqrt(pi))
                - pi * ( (1-sqrt(pi)) / (2*sqrt(pi)) )^4

"""

import time
from compnet.graphs import ConfigModelDegreeGraph
import numpy as np
import matplotlib.pyplot as plt

start = time.time()

# PART A: GENERATE INSTANCES AND MEASURE THE GIANT COMPONENT

n_samples = 20

# we pick equally spaced values of pi from 0.01 to 0.99
pi_values = np.linspace(0.01, 0.99, 25)

# we define the lists of means and standard deviations of giant component sizes
gcs_means, gcs_stds = [], []

# we iterate over values of the pi parameter
for pi in pi_values:
    print(f'Generating data for instances with pi={pi}')

    # we define the number of vertices of a graph
    N = 1_000

    # we define the degree distribution
    degree_dict = {1: 1-pi, 4: pi}

    # we generate a ConfigModelDegreeGraph object instance
    G = ConfigModelDegreeGraph(N=N, degree_dict=degree_dict)

    # initialize the list of giant component sizes for random instances given pi
    gcs_samples = []

    # for each value of pi, we sample 20 instances
    for inst in range(n_samples):

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
    gcs_stds.append(np.std(gcs_samples) / np.sqrt(n_samples))

# PART B: GENERATING THE PLOT FOR THE ANALYTICAL COMPUTATION OF THE
# GIANT COMPONENT SIZE

def gamma_func(pi):
    """
    This is the gamma function returning the probability that a vertex belongs
    to the giant component in the specific case of p1=1-pi and p4=pi for a
    configuration model random graph instance.

    :param pi: input parameter pi
    :returns gamma: the probability that a vertex belongs to the GC
    """

    def mu_func(pi):
        """
        This is the mu function returning the probability that an end vertex
        does not belong to the GC in the specific case of p1=1-pi and p4=pi for
        a configuration model random graph instance.

        :param pi: input parameter pi
        :returns gamma: the probability that an end vertex does not belong to
        the GC
        """
        return (1-np.sqrt(pi)) / (2*np.sqrt(pi))

    # we put a threshold where we know that the GC will appear
    if pi > 1. / 9.:
        return 1 - (1 - pi) * mu_func(pi) - pi * (mu_func(pi))**4
    else:
        return 0.

analytic_pi = np.linspace(0, 1, 200)
analytic_gamma = np.vectorize(gamma_func)(analytic_pi)

# generate the figure
fig = plt.figure()
plt.errorbar(pi_values, gcs_means, yerr=gcs_stds, fmt='.',
    label='GCS for $N=1000$ instances')
plt.plot(analytic_pi, analytic_gamma, label=r'$\gamma$ function')
plt.grid()
plt.legend()
plt.title('Behaviour of GCS in the configuration model')
plt.xlabel(r'$\pi$')
plt.ylabel(r'Relative giant component size')
plt.close()

# saving the figure
fig.savefig('assets/gcs.pdf', bbox_inches='tight')

stop = time.time()

print(f"Time elapsed: {(stop - start) / 60} mins")
