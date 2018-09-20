from __future__ import print_function

import numpy as np
import pint

from input_output import ureg
import aux_calcs

def firing_rates(dimension, tau_m, tau_s, tau_r, V_0_rel, V_th_rel, K, J, j,
                 nu_ext, K_ext):
    '''
    Returns vector of population firing rates in Hz.

    Parameters:
    -----------
    dimension: Quantity(int, 'dimensionless')
        number of populations
    tau_m: Quantity(float, 'millisecond')
        membrane time constant
    tau_s: Quantity(float, 'millisecond')
        synaptic time constant
    tau_r: Quantity(float, 'millisecond')
        refractory time
    V_0_rel: Quantity(float, 'millivolt')
        relative reset potential
    V_th_rel: Quantity(float, 'millivolt')
        relative threshold potential
    K: Quantity(np.ndarray, 'dimensionless')
        indegree matrix
    J: Quantity(np.ndarray, 'millivolt')
        effective connectivity matrix
    j: Quantity(float, 'millivolt')
        effective connectivity weight
    nu_ext: Quantity(float, 'hertz')
        firing rate of external input
    K_ext: Quantity(np.ndarray, 'dimensionless')
        numbers of external input neurons to each population

    Returns:
    --------
    Quantity(np.ndarray, 'hertz')
        array of firing rates of each population in hertz
    '''

    def rate_function(mu, sigma):
        """ calculates stationary firing rate with given parameters """
        return aux_calcs.nu0_fb433(tau_m, tau_s, tau_r, V_th_rel, V_0_rel, mu,
                                   sigma)

    def get_rate_difference(nu):
        """ calculate difference between new iteration step and previous one """
        mu = mean(nu, K, J, j, tau_m, nu_ext, K_ext)
        sigma = standard_deviation(nu, K, J, j, tau_m, nu_ext, K_ext)
        new_nu = np.array([x.magnitude for x in list(map(rate_function, mu,
                                                         sigma))])*ureg.Hz
        return -nu + new_nu

    # do iteration procedure, until stationary firing rates are found
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
    '''
    Calc mean inputs to populations as function of firing rates of populations

    Following Fourcaud & Brunel (2002)

    Parameters:
    -----------
    nu: Quantity(np.ndarray, 'hertz')
        firing rates of populations
    K: Quantity(np.ndarray, 'dimensionless')
        indegree matrix
    J: Quantity(np.ndarray, 'millivolt')
        effective connectivity matrix
    j: Quantity(float, 'millivolt')
        effective connectivity weight
    tau_m: Quantity(float, 'millisecond')
        membrane time constant
    nu_ext: Quantity(float, 'hertz')
        firing rate of external input
    K_ext: Quantity(np.ndarray, 'dimensionless')
        numbers of external input neurons to each population

    Returns:
    --------
    Quantity(np.ndarray, 'millivolt')
        array of mean inputs to each population in millivolt
    '''

    # contribution from within the network
    m0 = np.dot(K * J, nu) * tau_m
    # contribution from external sources
    m_ext = j * K_ext * nu_ext * tau_m
    # add them up
    m = m0 + m_ext
    return m


@ureg.wraps(ureg.mV, (ureg.Hz, ureg.dimensionless, ureg.mV, ureg.mV, ureg.s,
                      ureg.Hz, ureg.dimensionless))
def standard_deviation(nu, K, J, j, tau_m, nu_ext, K_ext):
    '''
    Calc standard devs of inputs to populations as function of firing rates

    Following Fourcaud & Brunel (2002)

    Parameters:
    -----------
    nu: Quantity(np.ndarray, 'hertz')
        firing rates of populations
    K: Quantity(np.ndarray, 'dimensionless')
        indegree matrix
    J: Quantity(np.ndarray, 'millivolt')
        effective connectivity matrix
    j: Quantity(float, 'millivolt')
        effective connectivity weight
    tau_m: Quantity(float, 'millisecond')
        membrane time constant
    nu_ext: Quantity(float, 'hertz')
        firing rate of external input
    K_ext: Quantity(np.ndarray, 'dimensionless')
        numbers of external input neurons to each population

    Returns:
    --------
    Quantity(np.ndarray, 'millivolt')
        array of standard dev of inputs to each population in millivolt
    '''
    # contribution from within the network to variance
    var0 = np.dot(K * J**2, nu) * tau_m
    # contribution from external sources to variance
    var_ext = j**2 * K_ext * nu_ext * tau_m
    # add them up
    var = var0 + var_ext
    # standard deviation is square root of variance
    std = np.sqrt(var)
    return std
