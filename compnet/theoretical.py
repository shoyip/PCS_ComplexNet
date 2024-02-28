import numpy as np

def relu(x):
  """
  Get the ReLU of an array.

  This function returns 0 if input is negative, and the input itself if it is
  positive, applied to each entry of the input array.

  :param x: Input array
  :returns: ReLU activated input array
  """
  return np.where(x>0, x, 0)

def get_theo_qcore_size(n_vertices, degree_distribution, q, function):
  """
  Get the size of the giant q-core in a configuration model graph with prescribed
  degree distribution.

  This function integrates the Ordinary Differential Equations that describe the
  evolution of a stochastic algorithm that for each timestep takes a node away
  from the graph, given that its degree is less than q.

  Each discrete timestep corresponds to a node being removed from the graph,
  hence we will get a giant q-core whenever the algorithm will have not yet
  exhausted the vertex set and only nodes of degree greater or equal than q
  will have remained.

  We will perform the integration of the ODE using the Runge-Kutta algorithm.
  The Ordinary Differential Equation takes up the form

    dp(t) / dt = f( p(t) )

  :param n_vertices: Number of vertices of the graph
  :param degree_distribution: List of degree probabilities in order
  :param q: The q number for which we want to find the q-core size
  :param function: The function on the RHS of the ODE
  :returns qcore_size, U: The integer size of the q-core and the evolution
    matrix of the ODE
  """

  # each timestep corresponds to a node removal given d < q
  h = 1. / n_vertices
  t = np.linspace(0., 1., n_vertices)
  n_variables = len(degree_distribution)
  U = np.zeros([n_vertices, n_variables])
  U[0] = np.array(degree_distribution)

  # the default value for the q-core size is 0
  # this means that we exhausted the vertex set and we did not find a q-core
  qcore_size = 0

  # perform Runge-Kutta integration
  for i in range(n_vertices-1):
    if (np.sum(U[i][q:]) == 0.):
      qcore_size = n_vertices - i
      break
    K_1 = function(t[i], U[i])*h
    K_2 = function(t[i]+h/2, U[i]+K_1/2)*h
    K_3 = function(t[i]+h/2, U[i]+K_2/2)*h
    K_4 = function(t[i]+h, U[i]+K_3)*h
    # added relu in order to bound values to positive numbers
    U[i+1] = relu(U[i] + (K_1 + 2*K_2 + 2*K_3 + K_4)/6)

  return qcore_size, U

def get_theo_gcomp_size(x, n_vertices):
    """
    TODO
    """
    return (1 - (1-x) * (1-np.sqrt(x)) / (2*np.sqrt(x)) - x * ((1 - np.sqrt(x))/(2*np.sqrt(x)))**4) * n_vertices
