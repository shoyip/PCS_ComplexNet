from compnet.graphs import ConfigModelDegreeGraph

degree_dict = {1: 0.3, 4: 0.7}

G = ConfigModelDegreeGraph(N=200, degree_dict=degree_dict)

edges = G.generate_graph()

print(edges)
