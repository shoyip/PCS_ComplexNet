"""
PROBLEM 5: INVERSE ISING MODEL
"""

from compnet.ising import IsingConfigModelDegreeGraph
import numpy as np
import matplotlib.pyplot as plt

def get_spin_configs(M, N, pi):
    degree_dict = {1: 1-pi, 4: pi}
    G = IsingConfigModelDegreeGraph(N=N, degree_dict=degree_dict)
    G.generate_graph()
    G.find_adjacency()
    spin_configs = []
    for m in range(M):
        G.generate_spins()
        _ = G.mcmc_wolff(T=4., mcmc_sweeps=500)
        spin_configs.append(np.array(G.spins))
    return np.array(spin_configs), G.adjacency

def get_correlation_ppv(correlation_matrix, adj, M):
    # find the indices of the off-diagonal lower triangular elements
    correlation_indices = np.array(np.tril_indices_from(correlation_matrix, k=-1)).T
    # find only the flattened i<j correlation terms
    correlations_f = correlation_matrix[np.tril_indices_from(correlation_matrix, k=-1)]
    # now we find the (flattened) indices sorted by correlation values
    sorted_correlation_indices_f = np.argsort(correlations_f)[::-1]
    # now we find the tuple ij values for the sorted correlations
    sorted_correlation_indices = list(map(tuple, correlation_indices[sorted_correlation_indices_f]))
    # find the ppv values
    ppv = np.cumsum([adj[indices_tuple] for indices_tuple in sorted_correlation_indices]) / np.arange(1, len(sorted_correlation_indices)+1)

    return ppv

def get_interaction_ppv(interaction, adj, M):
    # find the indices of the off-diagonal lower triangular elements
    interaction_indices = np.array(np.tril_indices_from(interaction, k=-1)).T
    # find only the flattened i<j interaction terms
    interactions_f = interaction[np.tril_indices_from(interaction, k=-1)]
    # now we find the (flattened) indices sorted by interaction values
    sorted_interaction_indices_f = np.argsort(interactions_f)[::-1]
    # now we find the tuple ij values for the sorted interactions 
    sorted_interaction_indices = list(map(tuple, interaction_indices[sorted_interaction_indices_f]))
    # find the ppv values
    ppv = np.cumsum([adj[indices_tuple] for indices_tuple in sorted_interaction_indices]) / np.arange(1, len(sorted_interaction_indices)+1)

    return ppv

def get_nmf_interactions(spin_configs, M):
    # let us compute the empirical magnetizations and correlations
    magnetizations = np.mean(spin_configs, axis=0).reshape(-1, 1)
    correlations = (spin_configs.T @ spin_configs) / M - magnetizations @ magnetizations.T
    
    # now let us find the interaction matrix as
    interactions = -np.linalg.inv(correlations)

    return interactions

if __name__ == "__main__":
    M = 1000
    N = 100
    pi = 0.7

    # Let's get the data
    spin_configs, adj = get_spin_configs(M, N, pi)
    mean_spin = np.mean(spin_configs, axis=0).reshape(-1, 1)
    correlation_matrix = (spin_configs.T @ spin_configs) / M - mean_spin @ mean_spin.T
    #print(correlation_matrix)
    #print(np.linalg.inv(correlation_matrix))

    # and compute the PPV
    corr_ppv = get_correlation_ppv(correlation_matrix, adj, M)

    # Now let us generate the interaction matrix in the NMF approx
    #interactions = get_nmf_interactions(spin_configs, M)
    interactions = -np.linalg.inv(correlation_matrix)
    #print(interactions)

    # and compute the PPV
    inter_ppv = get_interaction_ppv(interactions, adj, M)
    indices_of_ltm = np.array(np.tril_indices_from(interactions, k=-1)).T
    interactions_f = interactions[np.tril_indices_from(interactions, k=-1)]
    interactions_fsi = np.argsort(interactions_f)[::-1]
    #print(interactions_f)
    #print(interactions_fsi[0])
    

    # Generating the figure for the PPV
    fig = plt.figure()
    plt.title("Positive Predictive Values for inference parameters")
    plt.plot(corr_ppv, label='Correlation PPV', linestyle='-')
    plt.plot(inter_ppv, label='Interaction PPV', linestyle='--')
    plt.xlabel('# of correlation-ranked $(i, j)$ pairs considered')
    plt.ylabel('Ratio of positively predicted edges')
    plt.xscale('log')
    plt.legend()
    plt.grid()
    plt.close()

    fig.savefig('assets/ppv.pdf', bbox_inches='tight')

    # and group the correlation matrix element values depending on
    # their presence as edges in the adjacency matrix
    edge_Cvalues = []
    nonedge_Cvalues = []
    for i in range(N):
        for j in range(i):
            if adj[i, j] > 0:
                edge_Cvalues.append(correlation_matrix[i, j])
            else:
                nonedge_Cvalues.append(correlation_matrix[i, j])

    # and group the interaction matrix element values depending on
    # their presence as edges in the adjacency matrix
    edge_Jvalues = []
    nonedge_Jvalues = []
    for i in range(N):
        for j in range(i):
            if adj[i, j] > 0:
                edge_Jvalues.append(interactions[i, j])
            else:
                nonedge_Jvalues.append(interactions[i, j])

    # Generating the histogram for the C matrix
    fig = plt.figure()
    plt.hist(nonedge_Cvalues, label='$C_{ij}$ for nonedges')
    plt.hist(edge_Cvalues, label='$C_{ij}$ for edges')
    plt.xlabel('$C_{ij}$ values')
    plt.title('Histogram of correlation values for edges and nonedges')
    plt.legend()
    plt.grid()
    plt.close()

    fig.savefig('assets/nmf_chist.pdf', bbox_inches='tight')

    # Generating the histogram for the J matrix
    fig = plt.figure()
    plt.hist(nonedge_Jvalues, label='$J_{ij}$ for nonedges')
    plt.hist(edge_Jvalues, label='$J_{ij}$ for edges')
    plt.xlabel('$J_{ij}$ values')
    plt.title('Histogram of interaction values for edges and nonedges')
    plt.legend()
    plt.grid()
    plt.close()

    fig.savefig('assets/nmf_jhist.pdf', bbox_inches='tight')
