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

def autocorr(x):
    r2=np.fft.ifft(np.abs(np.fft.fft(x))**2).real
    c=(r2/x.shape-np.mean(x)**2)/np.std(x)**2
    return c[:len(x)//2]
