import time
from compnet.graphs import ConfigModelDegreeGraph
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def get_gcs_sample(N, pi):
    degree_dict = {1: 1-pi, 4: pi}
    G = ConfigModelDegreeGraph(N=N, degree_dict=degree_dict)
    G.generate_graph()
    G.find_neighbourhoods()
    
    gcs = G.get_giantcomponentsize() / N

    return gcs

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

if __name__ == "__main__":
    n_samples = 20
    pi_values = np.linspace(0.01, 0.99, 25)
    N_values = [100, 500, 1000]
    gcs_meas = {N: {'means': [], 'stds': []} for N in N_values}
    for N in N_values:
        gcs_values_means = []
        gcs_values_stds = []
        for pi in pi_values:
            print(f'Sampling for PI {pi:.2f}')
            gcs_samples = []
            for _ in range(n_samples):
                gcs_sample = get_gcs_sample(N, pi)
                print(gcs_sample)
                gcs_samples.append(gcs_sample)
            gcs_values_means.append(np.mean(gcs_samples))
            gcs_values_stds.append(np.std(gcs_samples) / np.sqrt(n_samples))
            #print('std', np.std(gcs_samples) / np.sqrt(n_samples))
        gcs_meas[N]['means'] = gcs_values_means
        gcs_meas[N]['stds'] = gcs_values_stds

    analytic_pi = np.linspace(0, 1, 200)
    analytic_gamma = np.vectorize(gamma_func)(analytic_pi)

    fig = plt.figure()
    for N in N_values:
        plt.errorbar(pi_values, gcs_meas[N]['means'], yerr=gcs_meas[N]['stds'], fmt='.', label=f'GCS measures for $N={N}$')
    plt.plot(analytic_pi, analytic_gamma, label=r'$\gamma$ function')
    plt.xlabel('$\pi$')
    plt.ylabel('Size of GCS')
    plt.title('Behaviour of GCS in the configuration model')
    plt.legend()
    plt.grid()
    plt.close()

    fig.savefig('assets/gcs.pdf', bbox_inches='tight')
