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
from .utils import _cache

import nnmt

ureg = nnmt.ureg


def delay_dist_matrix(network, freqs=None):
    '''
    Calcs matrix of delay distribution specific pre-factors at given freqs.

    Assumes lower boundary for truncated Gaussian distributed delays to be zero
    (exact would be dt, the minimal time step).

    Parameters
    ----------
    network : Network object
        The network for which to calcluate the delay distribution matrix. It
        needs to have the following items in :code:`network_params`:

            Delay : array_like
                Delay matrix in seconds
            Delay_sd : array_like
                Delay standard deviation matrix in seconds.
            delay_dist : {'none', 'truncated_gaussian', 'gaussian'}
                String specifying delay distribution.

        And the following items in :code:`analysis_params`:

            omegas : array_like, optional
                The considered angular frequencies , if `freqs` is not passed
                explicitely.

    freqs : array_like, optional
        The frequencies for which to calculate the delay distribution matrix in
        Hz.

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
def _delay_dist_matrix(Delay, Delay_sd, delay_dist, omegas):
    '''
    Calcs matrix of delay distribution specific pre-factors at given freqs.

    See Also
    --------
    delay_dist_matrix : Wrapper function with full documentation.
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
