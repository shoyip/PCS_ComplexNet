import numpy as np
from .graphs import ConfigModelDegreeGraph

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
