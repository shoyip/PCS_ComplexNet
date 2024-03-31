import numpy as np
import matplotlib.pyplot as plt
import time

def cavfx(cavity_fields, T):
    beta = 1. / T
    return 1. / 2. / beta * np.sum(np.log(
        np.cosh(beta * (cavity_fields + 1) / np.cosh(beta * (cavity_fields - 1)))
        ))

def mom1(x):
    return np.mean(x)

def mom2(x):
    return np.mean(x*x)

def mom3(x):
    return np.mean(x*x*x)

def popdyn_cavity(pi, T, L=10_000, max_iter=100_000):
    # define the excess degrees
    q1 = (1-pi) / (1+3*pi)
    q4 = (4*pi) / (1+3*pi)

    # define the population
    population = np.random.uniform(-1, 1, L)

    # define moment lists
    moms1, moms2, moms3 = [], [], []

    # iterate until moment convergence is reached
    for i in range(max_iter):
        # pick a degree with probabilities q_d+1
        d = np.random.choice([0, 3], p=[q1, q4])
        # choose a number d+1 of cavity fields
        fields_idx = np.random.randint(0, L, size=d+1)
        # we will input the last d cavity fields in the cavity function
        # the we will replace the 0-th cavity field with the output
        population[fields_idx[0]] = cavfx(population[fields_idx[1:]], T)

        #moms1.append(mom1(population))
        #moms2.append(mom2(population))
        #moms3.append(mom3(population))

#    plt.figure()
#    plt.plot(moms1, label='order 1')
#    plt.plot(moms2, label='order 2')
#    plt.plot(moms3, label='order 3')
#    plt.legend()
#    plt.show()

    return population

def popdyn_fields(population, pi, T, max_iter=100_000):
    # define the degrees
    p1 = 1-pi
    p4 = pi

    L = len(population)

    # define moment lists
    moms1, moms2, moms3 = [], [], []

    # iterate until moment convergence is reached
    for i in range(max_iter):
        # pick a degree with probabilities q_d+1
        d = np.random.choice([1, 4], p=[p1, p4])
        # choose a number d+1 of cavity fields
        fields_idx = np.random.randint(0, L, size=d+1)
        # we will input the last d cavity fields in the cavity function
        # the we will replace the 0-th cavity field with the output
        population[fields_idx[0]] = cavfx(population[fields_idx[1:]], T)

        #moms1.append(mom1(population))
        #moms2.append(mom2(population))
        #moms3.append(mom3(population))

#    plt.figure()
#    plt.plot(moms1, label='order 1')
#    plt.plot(moms2, label='order 2')
#    plt.plot(moms3, label='order 3')
#    plt.legend()
#    plt.show()

    return population

def make_pd_histo():
    nT = 5
    npi = 5
    L = 10_000
    max_iter = 100_000

    Ts = np.linspace(0.01, 4., nT)
    pis = np.linspace(0.01, 1., npi)

    pd_cav = np.zeros((nT, npi, L))
    pd_field = np.zeros((nT, npi, L))

    def get_pd_cavfield(T, pi):
        beta = 1. / T
        cavity_population = popdyn_cavity(pi=pi, T=T, max_iter=max_iter)
        fields_population = popdyn_fields(cavity_population, pi=pi, T=T, max_iter=max_iter)
        return cavity_population, fields_population

    figA, axsA = plt.subplots(nT, npi, figsize=(25,20), sharex='all')
    figB, axsB = plt.subplots(nT, npi, figsize=(25,20), sharex='all')
    for i, T in enumerate(Ts):
        for j, pi in enumerate(pis):
            start = time.time()
            #print(f'Computing mag value for T={T}, pi={pi}...')
            cavity_population, fields_population = get_pd_cavfield(T, pi)

            axsA[i, j].hist(cavity_population)
            axsB[i, j].hist(fields_population)
            axsA[i, j].title.set_text(f'$\pi$ {pi} T {T}')
            axsB[i, j].title.set_text(f'$\pi$ {pi} T {T}')
#            axsA[i, j].set_xlim([0,1])
#            axsB[i, j].set_xlim([0,1])

#            print(f'{i*nT+j}/{nT*npi} TEMP {T:.2f} PI {pi:.2f} MAG {pd_mags[i,j]:.2f}')
            stop = time.time()
            print(f'Took {(stop-start)//60:.0f}m{(stop-start)%60:.0f}s')

    plt.close()
    plt.close()

    figA.savefig('assets/pd_cavity.pdf', bbox_inches='tight')
    figB.savefig('assets/pd_fields.pdf', bbox_inches='tight')

#    np.save('data/pd_mags.npy', pd_mags)

if __name__ == "__main__":
    make_pd_histo()
#    pi = 0.5
#    T = 1.2
#    beta = 1. / T
#    cavity_population = popdyn_cavity(pi=pi, T=T, max_iter=100_000)
#    fields_population = popdyn_fields(cavity_population, pi=pi, T=T, max_iter=100_000)
#    mags = np.tanh(beta * fields_population)
#    print(np.mean(np.abs(mags)))
#    plt.figure()
#    plt.hist(mags)
#    plt.xlim([0, 1])
#    plt.show()
