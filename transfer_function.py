"""transfer_function.py: Calculates transfer function of leaky
integrate-and-fire neuron fed by filtered noise [1,2].

[1] Schuecker et al., "Modulated escape from a metastable state driven
by colored noise", Phys. Rev. E (2015)
[2] Schuecker et al., "Reduction of colored noise in excitable systems
to white noise and dynamic boundary conditions", arXiv:1410.8799v3
(2014)
[3] Bos et al., "Identifying anatomical origins of coexisting
oscillations in the cortical microcircuit", arXiv:1510.00642 (2015)


Authors: Jannis Schuecker, Moritz Helias, Hannah Bos

"""

import numpy as np
import mpmath
import siegert
from scipy.special import zetac
# remove fortran functions
# import fortran_functions as ff


def transfer_function_taylor(omega, params, mu, sigma):
    """
    Calculates transfer function according to Eq. 93 in [2]. The
    results in [3] were obtained with this expression and it is
    used throughout this package

    """

    # convert from ms to s
    taum = params['taum'] * 1e-3
    tauf = params['tauf'] * 1e-3
    taur = params['taur'] * 1e-3
    Vth = params['Vth']
    V0 = params['V0']

    # convert mu to absolute values (not relative to reset)
    mu += V0

    # for frequency zero the exact expression is given by the derivative of
    # f-I-curve
    if np.abs(omega - 0.) < 1e-15:
        return siegert.d_nu_d_mu_fb433(taum, tauf, taur, Vth, V0, mu, sigma)
    else:
        nu0 = siegert.nu_0(taum, taur, Vth, V0, mu, sigma)
        nu0_fb = siegert.nu0_fb433(taum, tauf, taur, Vth, V0, mu, sigma)
        x_t = np.sqrt(2.) * (Vth - mu) / sigma
        x_r = np.sqrt(2.) * (V0 - mu) / sigma
        z = complex(-0.5, complex(omega * taum))
        alpha = np.sqrt(2) * abs(zetac(0.5) + 1)
        k = np.sqrt(tauf / taum)
        A = alpha * taum * nu0 * k / np.sqrt(2)
        a0 = Phi_x_r(z, x_t, x_r)
        a1 = dPhi_x_r(z, x_t, x_r) / a0
        a3 = A / taum / nu0_fb * (-a1**2 + d2Phi_x_r(z, x_t, x_r) / a0)
        result = np.sqrt(2.) / sigma * nu0_fb / \
            complex(1., omega * taum) * (a1 + a3)

        return result


def transfer_function_shift(omega, params, mu, sigma):
    """
    Calculates transfer function according to $\tilde{n}$ in [1]. The
    expression is to first order equivalent to
    `transfer_function_taylor`. Since the underlying theory is
    correct to first order, the two expressions are exchangeable.
    We add it here for completeness, but it is not used in this package.

    """

    # convert from ms to s
    taum = params['taum'] * 1e-3
    tauf = params['tauf'] * 1e-3
    taur = params['taur'] * 1e-3
    Vth = params['Vth']
    V0 = params['V0']

    # convert mu to absolute value (not relative to reset)
    mu += V0

    # effective threshold and reset
    alpha = np.sqrt(2) * abs(zetac(0.5) + 1)
    Vth += sigma * alpha / 2. * np.sqrt(tauf / taum)
    V0 += sigma * alpha / 2. * np.sqrt(tauf / taum)

    # for frequency zero the exact expression is given by the derivative of
    # f-I-curve
    if np.abs(omega - 0.) < 1e-15:
        return siegert.d_nu_d_mu(taum, tauf, taur, Vth, V0, mu, sigma)
    else:
        nu = siegert.nu_0(taum, taur, Vth, V0, mu, sigma)

        x_t = np.sqrt(2.) * (Vth - mu) / sigma
        x_r = np.sqrt(2.) * (V0 - mu) / sigma
        z = complex(-0.5, complex(omega * taum))

        frac = dPhi_x_r(z, x_t, x_r) / Phi_x_r(z, x_t, x_r)

        return np.sqrt(2.) / sigma * nu / (1. + complex(0., complex(omega * taum))) * frac

# ______________________________________________________________________________
# Auxiliary functions

# using fortran_functions
# def Phi(a, x):
#     """
#     Calculates Phi(a,x) = exp(x**2/4)*U(a,x), where U(a,x) is the
#     parabolic cylinder function. Implementation uses the relation to
#     kummers function (Eq.19.12.1 and 13.1.32 in Handbook of
#     mathematical Functions, Abramowitz and Stegun, 1972, Dover
#     Puplications, New York). The latter are implemented in Fortran90.
#     """
#
#     fac1 = np.sqrt(np.pi) * 2**(-0.25 - 1 / 2. * a)
#     fac2 = np.sqrt(np.pi) * 2**(0.25 - 1 / 2. * a) * x
#     kummer1 = ff.kummers_function(0.5 * a + 0.25, 0.5, 0.5 * x**2)
#     term1 = kummer1 / mpmath.gamma(0.75 + 0.5 * a)
#     kummer2 = ff.kummers_function(0.5 * a + 0.75, 1.5, 0.5 * x**2)
#     term2 = kummer2 / mpmath.gamma(0.25 + 0.5 * a)
#     value = fac1 * term1 + fac2 * term2
#     value = complex(value.real, value.imag)
#     return value

def Phi_mpmath(z, x):
    """
    Calculates Phi(a,x) = exp(x**2/4)*U(a,x), where U(a,x) is the
    parabolic cylinder function. Implementation uses the mpmath
    functions. This is slower than the Fortran implementation `Phi`
    and not used in this package but added for completeness.
    """

    value = np.exp(0.25*x**2) * complex(mpmath.pcfu(z, -x))
    return value

def d_Phi(z, x):
    """
    First derivative of Phi using recurrence relations (Eq.: 12.8.9
    in http://dlmf.nist.gov/12.8)
    """

    return (1. / 2. + z) * Phi(z + 1, x)


def d_2_Phi(z, x):
    """
    Second derivative of Phi using recurrence relations (Eq.: 12.8.9
    in http://dlmf.nist.gov/12.8)
    """

    return (1. / 2. + z) * (3. / 2. + z) * Phi(z + 2, x)


def Phi_x_r(z, x, y):
    return Phi(z, x) - Phi(z, y)


def dPhi_x_r(z, x, y):
    return d_Phi(z, x) - d_Phi(z, y)


def d2Phi_x_r(z, x, y):
    return d_2_Phi(z, x) - d_2_Phi(z, y)
