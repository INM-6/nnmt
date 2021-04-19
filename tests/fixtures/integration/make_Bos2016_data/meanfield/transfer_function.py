"""transfer_function.py: Calculates transfer function of leaky
integrate-and-fire neuron fed by filtered noise after Schuecker et al.
Phys. Rev. E (2015).

Authors: Jannis Schuecker, Moritz Helias, Hannah Bos
"""

import numpy as np
import mpmath
import siegert
from scipy.special import zetac
import fortran_functions as ff


def Phi(a, x):
    """Calculates Phi(a,x) = exp(x**2/4)*U(a,x), where U(a,x) is the
    parabolic cylinder funciton. Implementation uses the relation to
    kummers function (Eq.19.12.1 and 13.1.32 in Handbook of
    mathematical Functions, Abramowitz and Stegun, 1972, Dover
    Puplications, New York)
    """

    fac1 = np.sqrt(np.pi) * 2**(-0.25 - 1 / 2. * a)
    fac2 = np.sqrt(np.pi) * 2**(0.25 - 1 / 2. * a) * x
    kummer1 = ff.kummers_function(0.5 * a + 0.25, 0.5, 0.5 * x**2)
    first_term = kummer1 / mpmath.gamma(0.75 + 0.5 * a)
    kummer2 = ff.kummers_function(0.5 * a + 0.75, 1.5, 0.5 * x**2)
    second_term = kummer2 / mpmath.gamma(0.25 + 0.5 * a)
    return fac1 * first_term + fac2 * second_term


def d_Phi(z, x):
    """First derivative of Phi using recurrence relations (Eq.: 12.8.9
    in http://dlmf.nist.gov/12.8)
    """

    return (1. / 2. + z) * Phi(z + 1, x)


def d_2_Phi(z, x):
    """Second derivative of Phi using recurrence relations (Eq.: 12.8.9
    in http://dlmf.nist.gov/12.8)
    """

    return (1. / 2. + z) * (3. / 2. + z) * Phi(z + 2, x)


def transfer_function(omega, params, mu, sigma):
    """Calculates transfer function of leaky-integrate and fire neuron
    model subjected to colored noise according to Eq. 93 in Schuecker et
    al. (2014) "Reduction of colored noise in excitable systems to white
    noise and dynamic boundary conditions" arXiv:1410.8799v3
    """

    taum = params['taum'] * 1e-3
    taus = params['tauf'] * 1e-3
    taur = params['taur'] * 1e-3
    V0 = 0.0
    dV = params['Vth'] - params['V0']
    if omega == 0.:
        return siegert.d_nu_d_mu_fb433(taum, taus, taur, dV, V0, mu, sigma)
    else:
        nu0 = siegert.nu_0(taum, taur, dV, V0, mu, sigma)
        nu0_fb = siegert.nu0_fb433(taum, taus, taur, dV, V0, mu, sigma)
        x_t = np.sqrt(2.) * (dV - mu) / sigma
        x_r = np.sqrt(2.) * (V0 - mu) / sigma
        z = complex(-0.5, complex(omega * taum))
        alpha = np.sqrt(2) * abs(zetac(0.5) + 1)
        k = np.sqrt(taus / taum)
        A = alpha * taum * nu0 * k / np.sqrt(2)

        def Phi_x_r(x, y):
            return Phi(z, x) - Phi(z, y)

        def dPhi_x_r(x, y):
            return d_Phi(z, x) - d_Phi(z, y)

        def d2Phi_x_r(x, y):
            return d_2_Phi(z, x) - d_2_Phi(z, y)

        a0 = Phi_x_r(x_t, x_r)
        a1 = dPhi_x_r(x_t, x_r) / a0
        a3 = A / taum / nu0_fb * (-a1**2 + d2Phi_x_r(x_t, x_r) / a0)
        result = np.sqrt(2.) / sigma * nu0_fb / \
            complex(1., omega * taum) * (a1 + a3)

        return result
