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
import matplotlib.pyplot as plt
import networkx as nx

for idx, pi in enumerate([0.1, 0.3, 0.7]):
    # we define a dictionary containing the degree distribution
    degree_dict = {1: 1-pi, 4: pi}
    
    # we define the graph object with N vertices and the given
    # degree distribution
    N = 200
    G = ConfigModelDegreeGraph(N=N, degree_dict=degree_dict)
    
    # sample a graph instance
    G.generate_graph()
    
    print("Graphs are stored as a list of edges.")
    print(G.edges)
    
    # find the neighbourhoods dictionary
    G.find_neighbourhoods()
    
    # import graph to networkx
    Gnx = nx.from_edgelist(G.edges)
    
    # draw random graph
    fig, ax = plt.subplots()
    nx.draw(Gnx, node_size=10, ax=ax)
    plt.close()
    
    fig.savefig(f'assets/rg{idx}.pdf', bbox_inches='tight')
