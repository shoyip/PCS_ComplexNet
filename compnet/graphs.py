import numpy as np

class ConfigModelGraph:
  """
  Instances of graphs generated with the configuration model given a certain
  degree sequence k.

  Parameters
  ----------
  N: int
    Number of vertices in the graph
  degree_sequence: list
    Degree sequence of the configuration model
  """
  def __init__(self, N, degree_sequence, allow_isolated=False):
    if ((sum(degree_sequence) % 2 != 0) & ~allow_isolated):
      raise ValueError("The degree sequence yields an odd number of stubs.")
    self.N = N
    self.degree_sequence = degree_sequence
    self.allow_isolated = allow_isolated
    self.stubs = self.generate_stubs()
    self.rng = np.random.default_rng()
  
  def generate_stubs(self):
    """
    Generate list of stubs, which are the id of each n-th vertex repeated
    k_n times.

    Returns
    -------
    stubs: list
      List of stubs
    """
    stubs = []
    for n, k in enumerate(self.degree_sequence):
      stubs += k * [n]
    return stubs

  def generate_graph(self, max_iter=100_000):
    """
    Generate a random instance of a graph with given vertices and degree
    sequence.

    Iterate through the stubs two-by-two and create edges. Check that edges are
    neither multiple edges nor are self-edges.
    """
    current_stubs = self.stubs
    current_edges = []
    success = 0
    for i in range(max_iter):
      if (len(current_stubs) == 0):
        success = 1
        break
      # find the indices and values of a candidate tuple of nodes
      trial_values = self.rng.choice(a=current_stubs,
                                     size=2)
      # if we have exhausted the stub set, exit the loop
      # if they are the same nodes, refuse the move
      # if they are already in the edge set, refuse the move
      # if none of these occur, add the candidate to the edge set
      if (trial_values[0] == trial_values[1]):
        continue
      elif ((id(trial_values) in map(id, current_edges)) |
            (id(np.flip(trial_values)) in map(id, current_edges))):
        continue
      else:
        current_edges.append(trial_values)
        for e in trial_values: current_stubs.remove(e)
    if (success == 0):
      raise ValueError("Graph not found.")
    return current_edges

class ConfigModelDegreeGraph(ConfigModelGraph):
  def __init__(self, N, degree_dict, allow_isolated=False,
               correct_odd=True):
    if (sum([degree_prob for degree_prob in degree_dict.values()]) != 1.):
      raise ValueError("Probability values are not correctly normalized.")
    for i in range(10_000):
      degree_sequence = np.random.choice(a=list(degree_dict.keys()),
                                              size=N,
                                              p=list(degree_dict.values()))
      if (correct_odd & sum(degree_sequence) % 2 == 0):
        break
      else:
        continue
    super(ConfigModelDegreeGraph, self).__init__(
        N=N,
        degree_sequence=degree_sequence,
        allow_isolated=allow_isolated)
