import numpy as np
import matplotlib.pyplot as plt

def show(lambdas, *, save_pdf=False, save_pdf_fname="lambdas.pdf"):
    """ Displays the lambda vectors in lambdas, also possible to save as pdf """
    for i, l in enumerate(lambdas):
        print(l)
        plt.plot(l, label="$\hat{\lambda}_"+str(i)+"$")
    plt.xticks(range(lambdas[0].shape[0]))
    plt.xlabel("$\hat{\lambda}_i$")
    plt.ylabel("weight")
    plt.legend()
    if save_pdf:
        plt.savefig(save_pdf_fname.pdf)

def linear(N, L):
    """ Returns L lambda vectors of size N with linear weights """
    lambdas = [np.interp(np.linspace(0., 1., N), [0., 1.], [lam, 1-lam]) for lam in np.linspace(1., .5, L+1)]
    lambdas = [l/l.sum() for l in lambdas]
    # remove flat
    lambdas = lambdas[:-1]
    return lambdas

def invpow(N, L):
    """ Returns L lambda vectors of size N with inverse power law weights """
    lambdas = [np.array([1/(x**n) for n in range(N)]) for x in np.linspace(1., 6., L+1)]
    lambdas.reverse()
    lambdas = [l/l.sum() for l in lambdas]
    # remove flat
    lambdas = lambdas[:-1]
    return lambdas

def sigmoid(N, L):
    """ Returns L lambda vectors of size N with sigmoid weights """
    sigmoid = lambda x: 1/(1+np.exp(-x))
    lspace = lambda x, N: np.linspace(-x, x, N)
    lambdas = [np.array([sigmoid(x) for x in lspace(y*2, N)])[::-1] for y in range(L+1)]
    # remove flat
    lambdas = lambdas[1:]
    lambdas = [l/l.sum() for l in lambdas]
    return lambdas

def sweetspot(N):
    """ Returns L=1 sweetspot lambda of size N described in the report """
    lambdas = [np.array([3/(4**(i+1)) for i in range(N-1)] + [.25**(N-1)])]
    return lambdas

def maxobf(N):
    """ Returns L=1 maxobf lambda of size N described in the report """
    lambdas = [np.ones(N)]
    lambdas = [l/l.sum() for l in lambdas]
    return lambdas
