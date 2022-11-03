# -*- coding: utf-8 -*-
'''
Calculations of network properties like the delay distribution matrix.

Functions
*********

.. autosummary::
    :toctree: _toctree/network_properties/

    delay_dist_matrix
    _delay_dist_matrix
    _lognormal_characteristic_function
    _mu_underlying_gaussian
    _sigma_underlying_gaussian

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
def _delay_dist_matrix(Delay, Delay_sd, delay_dist, omegas):
    '''
    Calcs matrix of delay distribution specific pre-factors at given freqs.

    Assumes lower boundary for truncated Gaussian distributed delays to be zero
    (exact would be dt, the minimal time step).

    Parameters
    ----------
    Delay : array_like
        Delay matrix (num_pop, num_pop) in seconds
    Delay_sd : array_like
        Delay standard deviation matrix (num_pop, num_pop) in seconds.
    delay_dist : {'none', 'truncated_gaussian', 'gaussian', 'lognormal'}
        String specifying delay distribution. For the lognormal distribution no
        closed form of the characteristic function is known. We therefore use
        the numeric approximation from :cite:t:`beaulieu2008`.
    omegas : array_like, optional
       The considered angular frequencies in 2*pi*Hz.

    Returns
    -------
    np.ndarray
        Matrix of delay distribution specific pre-factors at frequency omegas
        with shape (len(omegas), num_pop, num_pop).
    '''
    Omegas = np.array([np.ones(Delay.shape) * omega for omega in omegas])

    if delay_dist == 'none':
        return np.exp(- 1j * Omegas * Delay)

    elif delay_dist == 'truncated_gaussian':
        a0 = 0.5 * (1 + _erf((-Delay / Delay_sd + 1j * Omegas * Delay_sd)
                             / np.sqrt(2)))
        a1 = 0.5 * (1 + _erf((-Delay / Delay_sd) / np.sqrt(2)))
        b0 = np.exp(-0.5 * np.power(Delay_sd * Omegas, 2))
        b1 = np.exp(- 1j * Omegas * Delay)
        return (1.0 - a0) / (1.0 - a1) * b0 * b1

    elif delay_dist == 'gaussian':
        b0 = np.exp(-0.5 * np.power(Delay_sd * Omegas, 2))
        b1 = np.exp(- 1j * Omegas * Delay)
        return b0 * b1

    elif delay_dist == 'lognormal':
        # convert required mu and sigma to mean and var of underlying Gaussian
        Mu = _mu_underlying_gaussian(Delay, Delay_sd)
        Sigma = _sigma_underlying_gaussian(Delay, Delay_sd)

        # since integration for lognormal characteristic function is costly,
        # only integrate for unique combinations of mu, sigma, and omega

        # get unique combination of mu and sigma
        combs = np.vstack([Mu.flatten(), Sigma.flatten()]).T
        unique_combs = np.unique(combs, axis=0)

        # create unique combinations of mu, sigma, and omega
        combs_list = []
        for omega in omegas:
            combs_list.append(
                np.hstack([unique_combs,
                           omega * np.ones((len(unique_combs), 1))]
                          )
            )
        unique_combs = np.vstack(combs_list)

        omega_ids = {omega: i for i, omega in enumerate(omegas)}

        # calculate lognormal characteristic function for unique combinations
        # and put the results into the right places of the delay matrix
        D = np.ones(Omegas.shape, dtype='complex')
        for mu, sigma, omega in unique_combs:
            mask = np.where((Mu == mu) & (Sigma == sigma))
            D[omega_ids[omega]][mask] = (
                _lognormal_characteristic_function(omega, mu, sigma))

        return D


def _mu_underlying_gaussian(mu, sigma):
    """
    Computes the mean of the underlying Gaussian of a lognormal distribution.

    Parameters
    ----------
    mu : float or np.array
        Real mean of lognormal distribution.
    sigma : float or np.array
        Real standard deviation of lognormal distribution.

    Returns
    -------
    float of np.array
        Mean of underlying Gaussian.
    """
    return np.log(mu**2 / np.sqrt(mu**2 + sigma**2))


def _sigma_underlying_gaussian(mu, sigma):
    """
    Computes the std of the underlying Gaussian of a lognormal distribution.

    Parameters
    ----------
    mu : float or np.array
        Real mean of lognormal distribution.
    sigma : float or np.array
        Real standard deviation of lognormal distribution.

    Returns
    -------
    float of np.array
        Standard deviation of underlying Gaussian.
    """
    return np.sqrt(np.log(1 + sigma**2 / mu**2))


def _lognormal_characteristic_function(omega, mu, sigma):
    """
    Lognormal characteristic function

    Integration implementing :cite:t:`beaulieu2008` Eq. (6a) & (6b).

    Parameters
    ----------
    omega : float
        Frequency at which characteristic function is to be computed.
    mu : float
        Mean of underlying Gaussian.
    sigma : float
        Standard deviation of underlying Gaussian.

    Returns
    -------
    complex
        Characteristic function of specified lognormal distribution at omega.
    """
    # exp(mu) is used to include the non-zero mean of the underlying Gaussian
    # based on  Eq.(3) in Saberali, S. A., & Beaulieu, N. C. (2012, December).
    # New approximations to the lognormal characteristic function. In 2012 IEEE
    # Global Communications Conference (GLOBECOM) (pp. 2168-2172). IEEE.
    omega *= np.exp(mu)

    # Real part
    y1_0 = partial(_lognormal_integrand_real_A, omega=omega, sigma=sigma)
    y1_1 = partial(_lognormal_integrand_real_B, omega=omega, sigma=sigma)
    y1 = sint.quad(y1_0, 0, omega)[0] + sint.quad(y1_1, 0, 1/omega)[0]

    # Imaginary part
    y2_0 = partial(_lognormal_integrand_imag_A, omega=omega, sigma=sigma)
    y2_1 = partial(_lognormal_integrand_imag_B, omega=omega, sigma=sigma)
    y2 = sint.quad(y2_0, 0, omega)[0] + sint.quad(y2_1, 0, 1/omega)[0]

    return y1 + 1j * y2


def _lognormal_integrand_real_A(y, omega, sigma):
    """
    First part of :cite:t:`beaulieu2008` Eq. (6a) integrated from 0 to omega.
    """
    return np.cos(y) * _partial_integrand_A(y, omega, sigma)


def _lognormal_integrand_imag_A(y, omega, sigma):
    """
    First part of :cite:t:`beaulieu2008` Eq. (6b) integrated from 0 to omega.
    """
    return np.sin(y) * _partial_integrand_A(y, omega, sigma)


def _partial_integrand_A(y, omega, sigma):
    a = 1 / (y * sigma * np.sqrt(2 * np.pi))
    b = np.exp(-np.log(y/omega)**2 / (2 * sigma**2))
    return a * b


def _lognormal_integrand_real_B(y, omega, sigma):
    """
    Second part of :cite:t:`beaulieu2008` Eq. (6a) int from 0 to 1/omega.
    """
    return np.cos(1 / y) * _partial_integrand_B(y, omega, sigma)


def _lognormal_integrand_imag_B(y, omega, sigma):
    """
    Second part of :cite:t:`beaulieu2008` Eq. (6b) int from 0 to 1/omega.
    """
    return np.sin(1 / y) * _partial_integrand_B(y, omega, sigma)


def _partial_integrand_B(y, omega, sigma):
    a = 1 / (y * sigma * np.sqrt(2 * np.pi))
    b = np.exp(-1 * np.log(y*omega)**2 / (2 * sigma**2))
    return a * b
