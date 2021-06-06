# -*- coding: utf-8 -*-
'''
Calculations of network properties like the delay distribution matrix.

Functions
*********

.. autosummary::
    :toctree: _toctree/network_properties/
    
    delay_dist_matrix
    _delay_dist_matrix
    
    
Important Information
*********************

This is so cool!
'''


import numpy as np
from scipy.special import erf as _erf
from .utils import _cache

import lif_meanfield_tools as lmt

ureg = lmt.ureg


def delay_dist_matrix(network, freqs=None):
    """Wrapper for ``_delay_dist_matrix``."""
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
    lmt.utils._to_si_units(params)
    lmt.utils._strip_units(params)
    return _cache(network, _delay_dist_matrix, params, 'D')


@lmt.utils._check_positive_params
def _delay_dist_matrix(Delay, Delay_sd, delay_dist, omegas):
    '''
    Calcs matrix of delay distribution specific pre-factors at given freqs.

    Assumes lower boundary for truncated Gaussian distributed delays to be zero
    (exact would be dt, the minimal time step).

    Parameters
    ----------
    Delay : Quantity(np.ndarray, 's')
        Delay matrix.
    Delay_sd : Quantity(np.ndarray, 's')
        Delay standard deviation matrix.
    delay_dist : str
        String specifying delay distribution.
    omegas : float
        Frequency.

    Returns
    -------
    Quantity(nd.array, 'dimensionless')
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
