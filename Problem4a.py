import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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
    success = 0
    # repeat until we find a graph
    trial = 0
    while success == 0:
      trial += 1
      current_edges = []
      current_stubs = self.stubs.copy()
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

  def edges_to_neighbourhoods(self, edges):
    neighbourhoods = defaultdict(set)
    for edge in edges:
      edge = list(edge)
      neighbourhoods[edge[0]].add(edge[1])
      neighbourhoods[edge[1]].add(edge[0])
    return neighbourhoods

  def find_neighbourhoods(self):
    """
    Given a graph return the dictionary of neighbours for each vertex.
    Stores a dictionary of sets.
    """
    neighbourhoods = defaultdict(set)
    for edge in self.edges:
      edge = list(edge)
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
    current_edges = self.edges.copy()
    for iteration in range(10_000):
      nodes_to_be_deleted = set()
      current_neighbourhoods = self.edges_to_neighbourhoods(current_edges)
      for node, neighbourhood in current_neighbourhoods.items():
        # evaluate size of neighbourhood
        if len(neighbourhood) < q:
          nodes_to_be_deleted.add(node)

      # cycle through edges and delete the ones that are in the death note
      new_edges = []
      for edge in current_edges:
        keep_edge = len(edge.intersection(nodes_to_be_deleted))
        if keep_edge == 0: new_edges.append(edge)

      if len(current_edges) == 0:
        qcoresize = 0.
        break
      elif current_edges == new_edges:
        current_nodes = set.union(*current_edges)
        qcoresize = len(current_nodes)
        break

      current_edges = new_edges

    return qcoresize

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
        
class IsingConfigModelDegreeGraph(ConfigModelDegreeGraph):

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

    def mcmc_wolff(self,
                   T,
                   return_all=True,
                   mcmc_sweeps=1_000,
                   eq_sweeps=100,
                   sample_sweeps=50):
        def mcmc_wolff_sweep_mag(self, p_add):
            # update the neighbourhoods and the spin neighbourhoods
            self.find_neighbourhoods()
            self.find_spin_neighbourhoods()
            n = self.spin_neighbourhoods
            v = self.spins

            # pick randomly a node to initialize the wolff block
            idx_spin = np.random.randint(0, self.N)

            # define the wolff queue and visited spin lists
            queue = [idx_spin]
            visited = []

            # go on until queue is not exhausted for each spin block
            while len(queue) > 0:
                spin = self.spins[queue[-1]]
                n_v = self.spin_neighbourhoods[queue[-1]]
                n_i = self.neighbourhoods[queue[-1]]
                n_a = (n_v == spin)
                # append the current spin to the visited ones
                # and delete it from the queue
                visited.append(queue.pop(-1))
                # probabilistically add aligned spins to wolff block
                for ne_i, ne_a in zip(n_i, n_a):
                    if (np.random.random() < p_add) & (ne_a):
                        queue.append(ne_i)
                # make sure to remove the visited nodes from the queue
                queue = list(set(queue) - set(visited))

            # once the queue is exhausted, flip the visited spins
            for v_i in visited:
                self.spins[v_i] *= -1.

            return np.sum(self.spins)

        # refresh the ising spin configuration
        self.generate_spins()
        # define the parameters of the mcmc run
        beta = 1. / T
        p_add = 1. - np.exp(-2*beta)
        # iterate over the mcmc runs
        sweep_mags = []
        for mcmc_sweep in range(mcmc_sweeps):
            sweep_mag = mcmc_wolff_sweep_mag(self, p_add)
            if return_all:
                sweep_mags.append(sweep_mag)
            elif (mcmc_sweep%sample_sweeps) & (mcmc_sweep>eq_sweeps):
                sweep_mags.append(sweep_mag)
            else:
                continue
        if return_all:
            return np.array(sweep_mags) / self.N
        else:
            return np.mean(sweep_mags) / self.N

    def find_bp_cavity_fields(self, beta, eps=1e-1, max_iter=50):
        # for each graph instance there is one cavity fields fixed point
        # first of all let us find the adjacency matrix
        success = 0
        while success==0:
          self.find_adjacency()
          N = self.N
          adj = self.adjacency
          cavity_fields = np.random.normal(0., 1., size=(N, N)) * adj
          # let us iterate until we reach the fixed point
          for iteration in range(max_iter):
              new_cavity_fields = np.zeros((N, N))
              for i in range(N):
                  for j in range(N):
                      if (adj[i, j]==1.):
                          for k in range(N):
                              if (k!=j):
                                  new_cavity_fields[i, j] += cavity_function(cavity_fields[k, i], beta)
              err = np.abs(np.sum(cavity_fields - new_cavity_fields))
              #print(err)
              cavity_fields = new_cavity_fields
              if (err < eps):
                  print('fp reached')
                  success = 1
                  break
          if (success==0): print('fp not reached')

        # once we have found the fixed point, we save the cavity fields
        self.cavity_fields = cavity_fields

    def find_bp_fields(self, beta):
        N = self.N
        adj = self.adjacency
        # we then compute the local fields
        fields = np.zeros(N)
        for i in range(N):
            for j in range(N):
                if (adj[i, j]==1.):
                    fields[i] += cavity_function(self.cavity_fields[j, i], beta)
        self.fields = fields

    def find_bp_mag(self, beta):
        self.bp_mag = np.mean(np.tanh(beta * self.fields))

def autocorr(x):
    r2=np.fft.ifft(np.abs(np.fft.fft(x))**2).real
    c=(r2/x.shape-np.mean(x)**2)/np.std(x)**2
    return c[:len(x)//2]

def cavity_function(cavity_field, beta):
  return 1. / 2. / beta * np.log(np.cosh(beta * (cavity_field + 1)) / np.cosh(beta * (cavity_field - 1)))
  
nT = 5
npi = 10
pis = np.linspace(0.01, 0.99, npi)
Ts = np.linspace(0.1, 4., nT)
mags = np.zeros((nT, npi))
for i, pi in enumerate(pis):
  for j, T in enumerate(Ts):
    beta = 1. / T
    G = IsingConfigModelDegreeGraph(N=100, degree_dict={1: 1-pi, 4: pi})
    G.generate_graph()
    G.find_bp_cavity_fields(beta)
    G.find_bp_fields(beta)
    mag = np.mean(np.tanh(beta * G.fields))
    print(f'TEMP {T} PI {pi} MAG {mag}')
    mags[j, i] = mag
    
fig = plt.figure()
plt.pcolormesh(pis, Ts, np.abs(mags), shading='nearest')
plt.colorbar(label=r'$|m|$')
plt.xlabel('$\pi$')
plt.ylabel('$T$')
plt.xlim([0, 1])
plt.ylim([0, 4])
plt.close()

fig.savefig('assets/bp_trial.pdf')
