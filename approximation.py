import numpy as np
import elliptic_functions as ef
from matplotlib import pylab as plt


if __name__ == '__main__':

    if False:
        w = np.linspace(0.0, 2.0, num=1000)
        N = 5
        u = np.arccos(w.astype(complex))  # real in passband, imag in cutoff
        y = np.cos(N * u)  # purely real, except for roundoff
        k1 = 0.1  # arbitrary plotting kludge
        # compute key points
        vals = [1, 0, -1, 0]
        key_freqs = [np.cos(U/N*np.pi/2) for U in reversed(range(N+1))]
        key_vals  = [vals[U%4] for U in reversed(range(N+1))]

    else:
        w = np.linspace(0.0, 4.0, num=1000)
        k = 1 / 1.05
        N = 5
        L = N//2
        ui = np.array([(2*j+1)/N for j in range(L)])  # i = j + 1
        K = ef.complete_elliptic_integral(k)
        k1 = k**N * np.prod(ef.sne(ui,k)**4)
        K1 = ef.complete_elliptic_integral(k1)
        u = ef.acde(w.astype(complex),k)  # real C->S, imag S->N, complex N->D
        y = ef.cde(N*u,k1)  # purely real, except for roundoff

        # We only compute the passband key points, because the stopband key
        # points can be computed from the passband points, and because the
        # stopband contains NaN values that correspond to +/-Inf points.
        vals = [1, 0, -1, 0]
        key_freqs = [ef.cde(U/N,k) for U in reversed(range(N+1))]
        key_vals  = [vals[U%4] for U in reversed(range(N+1))]

    ep2 = 0.5

    # plt.plot(x, np.sqrt(1 / (1 + ep2 * np.real(y)**2)))
    plt.plot(w, np.real(y), '-')
    plt.plot(np.real(key_freqs), np.real(key_vals), 'o')
    plt.ylim(np.array([-1,1])/k1*1.5)
    plt.grid(True)
    plt.show()
