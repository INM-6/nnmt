# -*- coding: utf-8 -*-
'''
Calculations of network properties like the delay distribution matrix.

Functions
*********

.. autosummary::
    :toctree: _toctree/network_properties/

    delay_dist_matrix
    _delay_dist_matrix

'''
import numpy as np
from scipy.special import erf as _erf
import scipy.integrate as sint
from .utils import _cache

import nnmt

ureg = nnmt.ureg


def delay_dist_matrix(network, freqs=None):
    '''
    Calcs matrix of delay distribution specific pre-factors at given freqs.

    See :func:`nnmt.network_properties._delay_dist_matrix` for details.

    Parameters
    ----------
    network : Network object
        The network for which to calcluate the delay distribution matrix, with

        - ``network_params``: `Delay`, `Delay_sd`, `delay_dist`
        - ``analysis_params``: `freqs`, optional

    freqs : array_like, optional
        The frequencies for which to calculate the delay distribution matrix in
        Hz. Can alternatively be contained in the ``analysis_params`` of
        `network`.

    Returns
    -------
    np.ndarray
        Matrix of delay distribution specific pre-factors at frequency omegas.
    '''
    params = {}
    try:
        params['Delay'] = network.network_params['Delay']
        params['Delay_sd'] = network.network_params['Delay_sd']
        params['delay_dist'] = network.network_params['delay_dist']
        if freqs is None:
            params['omegas'] = network.analysis_params['omegas']
        else:
            params['omegas'] = np.atleast_1d(freqs) * 2 * np.pi
    except KeyError as param:
        raise RuntimeError(f'You are missing {param} for calculating the delay'
                           ' distribution matrix.')
    nnmt.utils._to_si_units(params)
    nnmt.utils._strip_units(params)
    return _cache(network, _delay_dist_matrix, params, 'D')


@nnmt.utils._check_positive_params
def _delay_dist_matrix(Delay, Delay_sd, delay_dist, omegas, 
                       integration_times=np.arange(1e-8, 1.0, 0.001)):
    '''
    Calcs matrix of delay distribution specific pre-factors at given freqs.

    Assumes lower boundary for truncated Gaussian distributed delays to be zero
    (exact would be dt, the minimal time step).

    Parameters
    ----------
    Delay : array_like
        Delay matrix in seconds
    Delay_sd : array_like
        Delay standard deviation matrix in seconds.
    delay_dist : {'none', 'truncated_gaussian', 'gaussian'}
        String specifying delay distribution.
    omegas : array_like, optional
       The considered angular frequencies in 2*pi*Hz.
    integration_times : array_like, optional
        Integration times used for numerical integration of the 
        Fourier-transform for distributions, for which no analytical solution 
        is available. 
        
        Default is np.arange(1e-8, 1.0, 0.001).
        
        The logarithmic pdf decays to zero for large delays. A delay of zero
        is not possible due to the logarithm.

    Returns
    -------
    np.ndarray
        Matrix of delay distribution specific pre-factors at frequency omegas.
    '''
    omegas = np.array([np.ones(Delay.shape) * omega for omega in omegas])

    if delay_dist == 'none':
        return np.exp(- 1j * omegas * Delay)

    elif delay_dist == 'truncated_gaussian':
        a0 = 0.5 * (1 + _erf((-Delay / Delay_sd + 1j * omegas * Delay_sd)
                             / np.sqrt(2)))
        a1 = 0.5 * (1 + _erf((-Delay / Delay_sd) / np.sqrt(2)))
        b0 = np.exp(-0.5 * np.power(Delay_sd * omegas, 2))
        b1 = np.exp(- 1j * omegas * Delay)
        return (1.0 - a0) / (1.0 - a1) * b0 * b1

    elif delay_dist == 'gaussian':
        b0 = np.exp(-0.5 * np.power(Delay_sd * omegas, 2))
        b1 = np.exp(- 1j * omegas * Delay)
        return b0 * b1

    elif delay_dist == 'lognormal':
        mu = mu_underlying_gaussian(Delay, Delay_sd)
        sigma = sigma_underlying_gaussian(Delay, Delay_sd)
        return lognormal_distribution_fourier(omegas,
                                              mu,
                                              sigma,
                                              integration_times)
        

def mu_underlying_gaussian(Delay, Delay_sd):
    return np.log(Delay**2 / np.sqrt(Delay**2 + Delay_sd**2))

def sigma_underlying_gaussian(Delay, Delay_sd):
    return np.sqrt(np.log(1 + Delay**2/Delay_sd**2))

def integrand_real(x, omega, mu_log, sigma_log):
    a1 = np.cos(np.outer(omega, x))
    a2 = 1 / (x * sigma_log * np.sqrt(2 * np.pi))
    a3 = np.exp(-1 * (np.log(x) - mu_log)**2 / (2 * sigma_log**2))
    return a1 * a2 * a3

def integrand_imag(x, omega, mu_log, sigma_log):
    a1 = np.sin(np.outer(omega, x))
    a2 = 1 / (x * sigma_log * np.sqrt(2 * np.pi))
    a3 = np.exp(-1 * (np.log(x) - mu_log)**2 / (2 * sigma_log**2))
    return a1 * a2 * a3

def lognormal_distribution_fourier(omega, mu, sigma, integration_times):
    y = np.zeros([omega.shape[0], *mu.shape], dtype=complex)
    # excitatory
    i, j = 0, 0
    y1 = integrand_real(integration_times, omega[:, i, j], mu[i, j], sigma[i, j])
    y1 = sint.simps(y1, integration_times)
    y2 = integrand_imag(integration_times, omega[:, i, j], mu[i, j], sigma[i, j])
    y2 = sint.simps(y2, integration_times)
    for i in range(mu.shape[0]):
        for j in range(0, mu.shape[1], 2):
            y[:, i, j] = y1-1j*y2 # e^*(-i wx)
    
    
    # inhibitory
    i, j = 1, 1
    y1 = integrand_real(integration_times, omega[:, i, j], mu[i, j], sigma[i, j])
    y1 = sint.simps(y1, integration_times)
    y2 = integrand_imag(integration_times, omega[:, i, j], mu[i, j], sigma[i, j])
    y2 = sint.simps(y2, integration_times)
    for i in range(mu.shape[0]):
        for j in range(1, mu.shape[1], 2):
            y[:, i, j] = y1-1j*y2
        
    
    return y