from __future__ import print_function

import numpy as np
import pint

from my_io import ureg
import aux_calcs

def firing_rates():
    return np.arange(8) * ureg.Hz

# @ureg.wraps(ureg.Hz, (ureg.ms, ureg.ms, ureg.ms, ureg.mV))
# def firing_rates(tau_m, tau_f, tau_r, dV):
#     '''Returns vector of population firing rates in Hz.'''
#     rate_function = lambda mu, sigma: aux_calcs.nu0_fb433(tau_m, tau_f, tau_r,
#                                                           dV, 0., mu, sigma)
#
#     @ureg.wraps(ureg.Hz, ureg.Hz)
#     def get_rate_difference(rates):
#         mu = self.get_mean(rates)
#         sigma = np.sqrt(self.get_variance(rates))
#         new_rates = np.array(list(map(rate_function, taum*fac*mu,
#                                  np.sqrt(taum)*fac*sigma)))
#         return -rates + new_rates
#
#     dt = 0.05
#     y = np.zeros((2, self.dimension))
#     eps = 1.0
#     while eps >= 1e-5:
#         delta_y = get_rate_difference(y[0])
#         y[1] = y[0] + delta_y*dt
#         epsilon = (y[1] - y[0])
#         eps = max(np.abs(epsilon))
#         y[0] = y[1]
#
#     return y[1]

@ureg.wraps(ureg.mV/ureg.s, (ureg.Hz, ureg.dimensionless, ureg.mV, ureg.mV,
                             ureg.Hz, ureg.dimensionless))
def mean(nu, K, W, J, nu_ext, K_ext):
    '''Returns vector of mean inputs to populations depending on
    the firing rates nu of the populations.

    Units as in Fourcoud & Brunel 2002, where current and potential
    have the same units, unit of output array: pA*Hz
    '''
    # contribution from within the network
    m0 = np.dot(K * W, nu)
    # contribution from external sources
    m_ext = J*K_ext*nu_ext
    m = m0 + m_ext
    return m


@ureg.wraps(ureg.mV**2/ureg.s, (ureg.Hz, ureg.dimensionless, ureg.mV, ureg.mV,
                                ureg.Hz, ureg.dimensionless))
def variance(nu, K, W, J, nu_ext, K_ext):
    '''Returns vector of variances of inputs to populations depending
    on the firing rates f of the populations.
    Units as in Fourcoud & Brunel 2002, where current and potential
    have the same units, unit of output array: pA*Hz
    '''
    # contribution from within the network
    var0 = np.dot(K * W**2, nu)
    # contribution from external sources
    var_ext = J**2 * K_ext * nu_ext
    var = var0 + var_ext

    return var
