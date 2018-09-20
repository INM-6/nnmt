from __future__ import print_function

import numpy as np
import pint

from input_output import ureg
import aux_calcs

# def firing_rates():
#     return np.arange(8) * ureg.Hz

# @ureg.wraps(ureg.Hz, (ureg.dimensionless, ureg.ms, ureg.ms, ureg.ms, ureg.mV,
#                       ureg.dimensionless, ureg.mV, ureg.mV, ureg.Hz,
#                       ureg.dimensionless))
def firing_rates(dimension, tau_m, tau_s, tau_r, V_0_rel, V_th_rel, K, J, j,
                 nu_ext, K_ext):
    '''Returns vector of population firing rates in Hz.'''
    rate_function = lambda mu, sigma: aux_calcs.nu0_fb433(tau_m, tau_s, tau_r,
                                                          V_th_rel, V_0_rel ,
                                                          mu, sigma)

    def get_rate_difference(nu):
        mu = mean(nu, K, J, j, tau_m, nu_ext, K_ext)
        sigma = standard_deviation(nu, K, J, j, tau_m, nu_ext, K_ext)
        new_nu = np.array([x.magnitude for x in list(map(rate_function, mu,
                                                         sigma))])*ureg.Hz
        return -nu + new_nu

    dt = 0.05
    y = np.zeros((2, int(dimension))) * ureg.Hz
    eps = 1.0
    while eps >= 1e-5:
        delta_y = get_rate_difference(y[0])
        y[1] = y[0] + delta_y*dt
        epsilon = (y[1] - y[0])
        eps = max(np.abs(epsilon.magnitude))
        y[0] = y[1]

    return y[1]

@ureg.wraps(ureg.mV, (ureg.Hz, ureg.dimensionless, ureg.mV, ureg.mV,
                      ureg.s, ureg.Hz, ureg.dimensionless))
def mean(nu, K, J, j, tau_m, nu_ext, K_ext):
    '''Returns vector of mean inputs to populations depending on
    the firing rates nu of the populations.

    Units as in Fourcoud & Brunel 2002, where current and potential
    have the same units, unit of output array: pA*Hz
    '''
    # contribution from within the network
    m0 = np.dot(K * J, nu) * tau_m
    # contribution from external sources
    m_ext = j * K_ext * nu_ext * tau_m
    m = m0 + m_ext
    return m


@ureg.wraps(ureg.mV, (ureg.Hz, ureg.dimensionless, ureg.mV, ureg.mV, ureg.s,
                      ureg.Hz, ureg.dimensionless))
def standard_deviation(nu, K, J, j, tau_m, nu_ext, K_ext):
    '''Returns vector of standard_deviations of inputs to populations depending
    on the firing rates f of the populations.
    Units as in Fourcoud & Brunel 2002, where current and potential
    have the same units, unit of output array: pA*Hz
    '''
    # contribution from within the network
    var0 = np.dot(K * J**2, nu) * tau_m
    # contribution from external sources
    var_ext = j**2 * K_ext * nu_ext * tau_m
    var = var0 + var_ext
    # standard deviation is square root of variance
    std = np.sqrt(var)
    return std
