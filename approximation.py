import numpy as np
import elliptic_functions as ef
from matplotlib import pylab as plt


if __name__ == '__main__':
  
    if False:
        x = np.linspace(0.0, 2.0)
        N = 3
        u = np.arccos(x.astype(complex))
        y = np.cos(N * u)
    else:
        x = np.linspace(0.0, 4.0, num=1000)
        k = 1 / 1.1
        N = 3
        L = N//2
        ui = np.array([(2*j+1)/N for j in range(L)])  # i = j + 1
        k1 = k**N * np.prod(ef.sne(ui,k)**4)
        K  = ef.complete_elliptic_integral(k)
        K1 = ef.complete_elliptic_integral(k1)
        u = ef.acde(x.astype(complex),k)
        y = ef.cde(N*u,k1)
        
    ep2 = 0.5

    plt.plot(x, np.sqrt(1 / (1 + ep2 * np.real(y)**2)))
    plt.grid(True)
    plt.show()
