"""
PROBLEM 1: GENERATION OF INSTANCES OF THE RANDOM GRAPH MODEL

This is the code that shows how we can use the implementation in compnet in
order to generate random instances of the configuration model, in particular in
the case of a given degree distribution with parametrized degree probabilities.

Differently from the general configuration model algorithm, this algorithm
discards self-edges and multiple edges between two nodes.

As shown in the code, instances of the graph are stored in the form of lists of
edges (two-values arrays), which is a manner of storing graphs in a more
efficient manner with respect to, say, adjacency matrices.

Another form we can choose to represent the data in, is the form of a dictionary
that contains for each node its neighbourhood.
"""

from compnet.graphs import ConfigModelDegreeGraph
import numpy as np

# we define a dictionary containing the degree distribution
pi = 0.3
degree_dict = {1: 1-pi, 4: pi}

# we define the graph object with N vertices and the given
# degree distribution
N = 20
G = ConfigModelDegreeGraph(N=N, degree_dict=degree_dict)

# sample a graph instance
G.generate_graph()

print("Graphs are stored as a list of edges.")
print(G.edges)

# find the neighbourhoods dictionary
G.find_neighbourhoods()

print("\nGraphs can be stored also as a dictionary of neighbourhoods.")
print(G.neighbourhoods)

# let us generate some graphs for visualization purposes
N = 100
for pi in [0.2, 0.6, 0.9]:
    degree_dict = {1: 1-pi, 4: pi}
    G = ConfigModelDegreeGraph(N=N, degree_dict=degree_dict)
    G.generate_graph()
    edges = np.array(G.edges)
    np.savetxt(f'assets/edges{pi:.1f}.csv', edges, delimiter=';')
