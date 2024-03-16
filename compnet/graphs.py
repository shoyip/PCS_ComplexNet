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
    self.nodes = np.arange(self.N)
    self.edges = None
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
    self.stubs = self.generate_stubs()
    current_stubs = self.stubs.copy()
    current_edges = []
    success = 0
    for i in range(max_iter):
      if (len(current_stubs) == 0):
        success = 1
        break
      # find the indices and values of a candidate tuple of nodes
      trial_values = set(self.rng.choice(a=current_stubs,
                                     size=2))
      # if we have exhausted the stub set, exit the loop
      # if they are the same nodes, refuse the move
      # if they are already in the edge set, refuse the move
      # if none of these occur, add the candidate to the edge set
      if (len(trial_values)==1):
        continue
      elif (trial_values in current_edges):
        continue
      else:
        current_edges.append(trial_values)
        for e in trial_values: current_stubs.remove(e)
    if (success == 0):
      raise ValueError("Graph not found.")
    self.edges = current_edges

  def get_components(self):
    """
    Get the components of the current graph instance.
    """
    if self.edges == None:
      raise ValueError("A graph must first be generated.")
    # transform the edge list in an edge array
    edges_arr = np.array(self.edges)
    # choose a random node
    nonvisited = np.array(self.nodes).copy()
    self.rng.shuffle(nonvisited)
    nonvisited = list(nonvisited)
    components = []
    for i_c in range(100_000):
      if len(nonvisited) == 0:
        break
      # for each component we initialize a list of visited nodes
      visited_c = set()
      nonvisited_c = [nonvisited[-1]]
      success_c = 0
      for i_nc in range(100_000):
        if len(nonvisited_c) == 0:
          success_c = 1
          break
        # visit the last nonvisited_c
        current_node = nonvisited_c[-1]
        current_neighbours = self.neighbourhoods[current_node]
        current_new_neighbours = current_neighbours - visited_c
        # remove the current node from the nonvisited and add it to visited
        visited_c.add(nonvisited_c.pop(-1))
        # now add the neighbours to the nonvisited
        nonvisited_c += current_new_neighbours
      if success_c == 0:
        continue
      else:
        for visited_node in visited_c: nonvisited.remove(visited_node)
        components.append(visited_c)
    return components

  def get_giantcomponentsize(self):
    """
    Get the size of the giant component for the current graph instance.
    """
    if self.edges == None:
      raise ValueError("A graph must first be generated.")
    components = self.get_components()
    component_sizes = [len(component) for component in components]
    max_component_size = max(component_sizes)
    return max_component_size

  def find_neighbourhoods(self):
    """
    Given a graph return the dictionary of neighbours for each vertex.
    Stores a dictionary of sets.
    """
    neighbourhoods = defaultdict(set)
    for edge in self.edges:
      neighbourhoods[edge[0]].add(edge[1])
      neighbourhoods[edge[1]].add(edge[0])
    self.neighbourhoods = dict(neighbourhoods)

  def get_degrees(self, neighbourhoods):
    """
    Given a neighbourhood dictionary, compute the degree for each vertex.
    """
    degrees = {k: len(v) for k, v in neighbourhoods.items()}
    return degrees

  def get_biggest_qcoresize(self, q):
    """
    Get the size of the biggest qcore for the current graph instance.
    """
    current_neighbourhoods = self.neighbourhoods.copy()
    # fictitious values to initialize the iteration
    current_qcoresize = 1
    previous_qcoresize = 0
    while (current_qcoresize > previous_qcoresize):
      current_degrees = self.get_degrees(current_neighbourhoods)
      previous_qcoresize = len(current_neighbourhoods)
      current_neighbourhoods = {k: v
      				for k, v
      				in current_neighbourhoods.items()
      				if current_degrees[k] >= q}
      current_qcoresize = len(current_neighbourhoods)
    return current_qcoresize

  def generate_spins(self):
    """
    Generate a random configuration of spin values.
    """
    self.spins = np.random.randint(low=0, high=2, size=self.N) * 2 - 1

  def find_spin_neighbourhoods(self):
    """
    Find and store the neighbourhood spin values of each spin.
    """
    self.spin_neighbourhoods = {k: [self.spins[e] for e in v] for k, v in self.neighbourhoods.items()}
  
  def find_adjacency(self):
    """
    Find and store the adjacency matrix.
    """
    adj = np.zeros(shape=(self.N, self.N))
    for i, j in self.edges:
      adj[i, j] = 1.
    self.adjacency = adj + adj.T

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
