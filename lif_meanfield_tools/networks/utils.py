import numpy as np
from scipy.special import erf as _erf

import lif_meanfield_tools as lmt


ureg = lmt.ureg


def delay_dist_matrix(network):
    params = {}
    try:
        params['Delay'] = network.network_params['Delay']
        params['Delay_sd'] = network.network_params['Delay_sd']
        params['delay_dist'] = network.network_params['delay_dist']
        params['omegas'] = network.analysis_params['omegas']
    except KeyError as param:
        raise RuntimeError(f'You are missing {param} for calculating the delay'
                           ' distribution matrix.')
    lmt.utils._to_si_units(params)
    lmt.utils._strip_units(params)
    return _delay_dist_matrix(**params)


@lmt.utils._check_positive_params
def _delay_dist_matrix(Delay, Delay_sd, delay_dist, omegas):
    '''
    Calcs matrix of delay distribution specific pre-factors at given freqs.

    Assumes lower boundary for truncated Gaussian distributed delays to be zero
    (exact would be dt, the minimal time step).

    We had to define the subfunctions ddm_none, ddm_tg and ddm_g, because one
    cannot pass a string to a function decorated with ureg.wraps. So, that is
    how we bypass this issue. It is not very elegant though.

    Parameters:
    -----------
    Delay: Quantity(np.ndarray, 's')
        Delay matrix.
    Delay_sd: Quantity(np.ndarray, 's')
        Delay standard deviation matrix.
    delay_dist: str
        String specifying delay distribution.
    omegas: float
        Frequency.

    Returns:
    --------
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
