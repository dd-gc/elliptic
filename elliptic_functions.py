"""
elliptic.py: Complex Jacobi Elliptic Function implementation.

This code is a Python implementation of some of the MATLAB code made available at
https://rutgers.box.com/shared/static/haot98kjm5k7a9151jaeid41qiksw9uw
written by:
Sophocles J. Orfanidis
ECE Department, Rutgers University
94 Brett Road, Piscataway, NJ 08854-8058, USA
https://www.ece.rutgers.edu/orfanidis

Just the code necessary for elliptic filter design was implemented.
"""

import numpy as np
import scipy as sp


def sym_fmod(v, x):
    """
    Function used to reduce values to a fundamental region.

    Based on the Rutgers srem function, with MATLAB rem equivalent to numpy.fmod.

    Parameters
    ----------
    v : array of float
        Vector of values periodic with period x.
    x : float
        Period of fundamental region.

    Returns
    -------
    array of float
        Values of v put into [x/2, x/2].

    """
    z = np.fmod(v, x)
    return z - x*np.sign(z)*(np.abs(z) > x/2)


def landen_decreasing_moduli(k, M=None):
    """
    Returns a series of Landen modulus transformations.

    Parameters
    ----------
    k : float, 0 <= k < 1
        Elliptic modulus.
    M : int or float, optional
        If an integer is specified, it sets the number of transformations.
        If a float is specified, it sets the maximum size of the last
           transformation before the transformations are truncated.
        The default is None, and sets M to the machine epsilon.

    Returns
    -------
    array of float
        The Landen transformations of the original modulus.

    """
    assert k < 1, 'modulus k must be less than 1'

    if M is None:
        M = np.finfo(float).eps  # machine precision

    if type(M) is int:
        # number of terms has been specified
        moduli = np.zeros((M,), dtype=float)

        for i in range(M):
            k = (k / (1 + np.sqrt(1-k*k)))**2
            moduli[i] = k

        return moduli

    else:
        # a tolerance has been specified for moduli
        moduli = np.zeros((16,), dtype=float)

        i = 0
        while i < len(moduli):
            k = k = (k / (1 + np.sqrt(1-k*k)))**2
            moduli[i] = k
            i += 1
            if k < M:
                break

        return moduli[:i]


def complete_elliptic_integral(k, M=None):
    """
    Compute the complete elliptic integral for a given modulus k.

    Other libraries call m = k^2 the modulus, but k is the argument here.

    K = complete_elliptic_integral(k) returns values very close to
    K = scipy.special.ellipk(k*k).

    Parameters
    ----------
    k : float
        Elliptic modulus.
    M : int or float, optional
        This is the value passed to landen_decreasing_moduli() to set accuracy.
        The default is None.

    Returns
    -------
    float
        The complete elliptic integral of modulus k.

    """
    moduli = landen_decreasing_moduli(k, M)

    return (1 + moduli).prod() * np.pi/2


def ellipk2(k, M=None):
    """
    Compute complete elliptic integrals for modulus k and k' = sqrt(1-k**2)

    Parameters
    ----------
    k : float
        Elliptic modulus.
    M : int or float, optional
        This is the value passed to landen_decreasing_moduli() to set accuracy.
        The default is None.

    Returns
    -------
    K : float
        Complete elliptic integral of k.
    Kp : float
        Complete elliptic integral of k'.

    """
    if k == 0:
        K = np.pi / 2
        Kp = np.inf
    elif k == 1:
        K = np.inf
        Kp = np.pi / 2
    else:
        K = complete_elliptic_integral(k, M)
        kp = np.sqrt(1 - k*k)
        Kp = complete_elliptic_integral(kp, M)

    return K, Kp


def cde(u, k, M=None):
    """
    Compute the Jacobi elliptic function cd of parameter u and modulus k.

    The parameter u is normalized to the quarter period K, and can be complex.
    For real values of u, cde(u,k) gives values very close to:
        K = scipy.special.ellipk(k*k)
        sn, cn, dn, ph = scipy.special.ellipj(K*u, m)  # scale u!
        cd = cn / dn
    From the original MATLAB documentation:
        the ratio R=K'/K determines the pattern
        of zeros and poles of the cd function      N ---- D(pole)    u=j*R ---- u=1+j*R
        within the SCDN fundamental rectangle,     |      |             |        |
        the pole at corner D is u = 1+j*R,         |      |             |        |
        the zero at corner C is u = 1              S ---- C(zero)      u=0 ---- u=1

        mappings around the C -> S -> N -> D path:
             C -> S, 0<=t<=1, u = 1-t    ==>    0 <= w <= 1     (passband)
             S -> N, 0<=t<=1, u = j*t*R  ==>    1 <= w <= 1/k   (transition)
             N -> D, 0<=t<=1, u = t+j*R  ==>  1/k <= w <= Inf   (stopband)

    Parameters
    ----------
    u : float or complex
        Normalized parameter of the cd function.
    k : float
        Elliptic modulus.
    M : int or float, optional
        This is the value passed to landen_decreasing_moduli() to set accuracy.
        The default is None.

    Returns
    -------
    w : real or complex
        Jacobi elliptic function cd(u,k).

    """
    moduli = landen_decreasing_moduli(k, M)

    w = np.cos(u * np.pi/2)

    for i in range(len(moduli)):
        w = (1 + moduli[-1-i]) * w / (1 + moduli[-1-i]*w*w)

    return w


def acde(w, k, M=None):
    """
    Computes the inverse of Jacobi elliptic function cd.

    If w = cde(u,k), then u1 = acde(w,k), u1 is approximately equal to u,
    except that the result is put in the fundamental region, 0 <= Re{u1} <= 2,
    -R <= Im{u1} <= +R, with R=K'/K (ratio of complete elliptic integrals)

    Parameters
    ----------
    w : float or complex
        Complex argument.
    k : float
        Elliptic modulus.
    M : int or float, optional
        This is the value passed to landen_decreasing_moduli() to set accuracy.
        The default is None.

    Returns
    -------
    u : complex
        The requested inverse.

    """
    moduli = landen_decreasing_moduli(k, M)

    for i in range(len(moduli)):
        if i == 0:
            v1 = k
        else:
            v1 = moduli[i-1]

        w = w / (1 + np.sqrt(1 - w*w * v1*v1)) * 2 / (1 + moduli[i])

    u = 2 / np.pi * np.arccos(w.astype(complex))  # gives a positive Re(u)

    # TODO: The MATLAB replaces u with zero for values where w is 1.0
    #       But where w is 1.0 to start, the transformation changes it to about 1-eps.

    # The result is not unique, reduce to fundamental region
    # Since Re(u) > 0, 0<Re(u)<2, and -R<Im(u)<R
    K, Kp = ellipk2(k)
    R = Kp / K
    # We changed the sign on the imaginary part so that as w goes from zero to
    # infinity, u maps from C to S to N to D.  Note that from N to D, imag(u)
    # alternates between +/-R due to roundoff.
    u = sym_fmod(np.real(u), 4) - 1j * sym_fmod(np.imag(u), 2*R)

    return u


def sne(u, k, M=None):
    """
    Return Jacobi elliptic function sn of parameter u and modulus k.

    Parameters
    ----------
    u : float or complex
        Parameter of sn function.
    k : float
        Elliptic modulus.
    M : int or float, optional
        This is the value passed to landen_decreasing_moduli() to set accuracy.
        The default is None.

    Returns
    -------
    w : float or complex
        sn(u,k).

    """
    moduli = landen_decreasing_moduli(k, M)
    
    w = np.sin(u * np.pi/2)

    for i in range(len(moduli)):
        w = (1 + moduli[-1-i]) * w / (1 + moduli[-1-i]*w*w)

    return w


def asne(w, k, M=None):
    """
    Returns inverse of Jacobi elliptic function sn.
    
    If w = sne(u,k), u1 = asne(w,k) is approximately u.

    Parameters
    ----------
    w : float or complex
        Parameter.
    k : float
        Elliptic modulus.
    M : int or float, optional
        This is the value passed to landen_decreasing_moduli() to set accuracy.
        The default is None.

    Returns
    -------
    complex
        The inverse value u.

    """
    return 1 - acde(w, k, M)


if __name__ == '__main__':

    # set the parameters of evaluation
    # ellipk is restricted to real values, unlike cde
    k = 0.9
    u = np.array([0.9, 1.0, 1.1])
    # u = 0.3

    # compare complete elliptic integral
    K_me = complete_elliptic_integral(k)
    m = k*k
    K_them = sp.special.ellipk(m)  # takes m=k^2 instead of k!

    # compare evaluation of cd
    cde_me = cde(u, k)
    sn, cn, dn, ph = sp.special.ellipj(K_them*u, m)  # scale u!
    cde_them = cn / dn

    # compute the inverse
    u_hat = acde(cde_me, k)
    
    # compute sn
    sne_me = sne(u, k)
    u_hat2 = asne(sne_me, k)  # TODO: not sure this is working correctly
