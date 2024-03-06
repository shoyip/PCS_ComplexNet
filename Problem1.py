from compnet.graphs import ConfigModelDegreeGraph

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
