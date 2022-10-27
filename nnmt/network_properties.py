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
from functools import partial

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
                       integration_x=np.arange(1e-8, 1.0, 0.001)):
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
    delay_dist : {'none', 'truncated_gaussian', 'gaussian', 'lognormal'}
        String specifying delay distribution.
        `Note`: For the lognormal distribution no closed form characteristic 
        function is known. We therefore use the numeric approximation from
        Beaulieu 2008. Fast convenient numerical computation of lognormal 
        characteristic functions. IEEE Transactions on communications, 56, 3  
    omegas : array_like, optional
       The considered angular frequencies in 2*pi*Hz.
    # integration_x : array_like, optional
    #     Integration times used for numerical integration of the 
    #     lognormal distribution, for which no analytical solution 
    #     is available.
        
    #     Default is np.arange(1e-8, 1.0, 0.001).

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
        # TODO check that delay mean cannot be negative
        mu = mu_underlying_gaussian(Delay, Delay_sd)
        sigma = sigma_underlying_gaussian(Delay, Delay_sd)
        return lognormal_distribution_beaulieu(omegas, mu,
                                              sigma,
                                              integration_x)
        
def mu_underlying_gaussian(Delay, Delay_sd):
    return np.log(Delay**2 / np.sqrt(Delay**2 + Delay_sd**2))

def sigma_underlying_gaussian(Delay, Delay_sd):
    return np.sqrt(np.log(1 + Delay**2/Delay_sd**2))

def lognormal_integrand_0(y, omega, sigma_log, part='real'):
    """
    part : ['real' or 'imag']
        determines whether the real or imaginary part is computed
    
    Integrated from 0 to omega   
    """
    if part == 'real': a1 = np.cos(1 / y)
    elif part == 'imag': a1 = np.sin(1 / y)
    a2 = 1 / (y * sigma_log * np.sqrt(2 * np.pi))
    a3 = np.exp(-1 * np.log(y/omega)**2 / (2 * sigma_log**2))
    return a1 * a2 * a3

def lognormal_integrand_1(y, omega, sigma_log, part='real'):
    """Integrated from 0 to 1/omega"""
    if part == 'real': a1 = np.cos(1 / y)
    elif part == 'imag': a1 = np.sin(1 / y)
    a2 = 1 / (y * sigma_log * np.sqrt(2 * np.pi))
    a3 = np.exp(-1 * np.log(y*omega)**2 / (2 * sigma_log**2))
    return a1 * a2 * a3

def lognormal_distribution_beaulieu(omega, mu, sigma, x):
    y = np.zeros([omega.shape[0], *mu.shape], dtype=complex)

    # Excitatory (i=0, j=0) and Inhibitory (i=1, j=1)
    for i, j in zip([0, 1], [0, 1]):
        # exp(mu) used to include the mean of the underlying Gaussian
        # based on Beaulieu et al 2012 Eq.(3)
        w_vector = omega[:, i, j] * np.exp(mu[i, j])
        s = sigma[i, j]

        for k, w in enumerate(w_vector):
            # Integration from Beaulieu 2008 Eq. 6a & 6b
        
            # Real part
            y1_0 = partial(lognormal_integrand_0, omega=w, sigma_log=s, part='real')
            y1_1 = partial(lognormal_integrand_1, omega=w, sigma_log=s, part='real')
            y1 = sint.quad(y1_0, 0, w)[0] + sint.quad(y1_1, 0, 1/w)[0]
           
            # Imaginary part
            y2_0 = partial(lognormal_integrand_0, omega=w, sigma_log=s, part='imag')
            y2_1 = partial(lognormal_integrand_1, omega=w, sigma_log=s, part='imag')
            y2 = sint.quad(y2_0, 0, w)[0] + sint.quad(y2_1, 0, 1/w)[0]

            # Final result
            for i in range(mu.shape[0]):
                for j in range(0, mu.shape[1], 2):
                    y[k, i, j] = y1-1j*y2      
    
    return y