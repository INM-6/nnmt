"""
Collection of functions for LIF neurons with delta synapses.

Network Functions
*****************

.. autosummary::
    :toctree: _toctree/lif/

    firing_rates
    mean_input
    std_input

Parameter Functions
*******************

.. autosummary::
    :toctree: _toctree/lif/

    _firing_rates
    _firing_rates_for_given_input
    _mean_input
    _std_input
    _derivative_of_firing_rates_wrt_mean_input

"""

import numpy as np
from scipy.special import (
    erf as _erf,
    erfcx as _erfcx,
    dawsn as _dawsn,
    roots_legendre as _roots_legendre
    )
from scipy.integrate import quad as _quad

from . import _general
from .. import _solvers
from ..utils import (_cache,
                     _check_positive_params)


_prefix = 'lif.delta.'


def firing_rates(network, **kwargs):
    """
    Calculates stationary firing rates for delta shaped PSCs.

    See :func:`nnmt.lif.delta._firing_rates` for full documentation.

    Parameters
    ----------
    network : nnmt.models.Network or child class instance.
        Network with the network parameters listed in the docstring of
        :func:`nnmt.lif.delta._firing_rates`.
    kwargs
        For additional kwargs regarding the fixpoint iteration procedure see
        :func:`nnmt._solvers._firing_rate_integration`.

    Returns
    -------
    np.array
        Array of firing rates of each population in Hz.
    """
    list_of_params = [
        'J', 'K',
        'V_0_rel', 'V_th_rel',
        'tau_m', 'tau_r',
        'K_ext', 'J_ext',
        'nu_ext',
        ]

    try:
        params = {key: network.network_params[key] for key in list_of_params}
    except KeyError as param:
        raise RuntimeError(
            f"You are missing {param} for calculating the firing rate!\n"
            "Have a look into the documentation for more details on 'lif' "
            "parameters.")

    params.update(kwargs)

    return _cache(network, _firing_rates, params, _prefix + 'firing_rates',
                  'hertz')


def _firing_rates(J, K, V_0_rel, V_th_rel, tau_m, tau_r, J_ext, K_ext, nu_ext,
                  **kwargs):
    """
    Calculation of firing rates for delta PSCs.

    See :func:`nnmt._solvers._firing_rate_integration` for integration
    procedure.

    Uses :func:`nnmt.lif.delta._firing_rates_for_given_input`.

    Parameters
    ----------
    J : np.array
        Weight matrix in V.
    K : np.array
        Indegree matrix.
    V_0_rel : [float | 1d array]
        Relative reset potential in V.
    V_th_rel : [float | 1d array]
        Relative threshold potential in V.
    tau_m : [float | 1d array]
        Membrane time constant of post-synatic neuron in s.
    tau_r : [float | 1d array]
        Refractory time in s.
    J_ext : np.array
        External weight matrix in V.
    K_ext : np.array
        Numbers of external input neurons to each population.
    nu_ext : 1d array
        Firing rates of external populations in Hz.
    kwargs
        For additional kwargs regarding the fixpoint iteration procedure see
        :func:`nnmt._solvers._firing_rate_integration`.

    Returns
    -------
    np.array
        Array of firing rates of each population in Hz.
    """
    firing_rate_params = {
        'V_0_rel': V_0_rel,
        'V_th_rel': V_th_rel,
        'tau_m': tau_m,
        'tau_r': tau_r,
        }
    input_params = {
        'J': J,
        'K': K,
        'tau_m': tau_m,
        'J_ext': J_ext,
        'K_ext': K_ext,
        'nu_ext': nu_ext,
        }

    input_funcs = [_general._mean_input, _general._std_input]

    return _solvers._firing_rate_integration(_firing_rates_for_given_input,
                                             firing_rate_params,
                                             input_funcs,
                                             input_params, **kwargs)


@_check_positive_params
def _firing_rates_for_given_input(mu, sigma, V_0_rel, V_th_rel, tau_m, tau_r):
    """
    Calculates stationary firing rate for delta shaped PSCs.

    Implementation of formula by Siegert for the mean-first-passage time
    :cite:p:`siegert1951`, found for example in Appendix A, Eq. A7 of
    :cite:t:`amit1997`.

    Parameters
    ----------
    mu : [float | 1d array]
        Mean input to population of neurons.
    sigma : [float | 1d array]
        Standard deviation of input to population of neurons.
    V_0_rel : [float | 1d array]
        Relative reset potential in V.
    V_th_rel : [float | 1d array]
        Relative threshold potential in V.
    tau_m : [float | 1d array]
        Membrane time constant of post-synatic neuron in s.
    tau_r : [float | 1d array]
        Refractory time in s.

    Returns
    -------
    [float | np.array]
        Firing rates in Hz.
    """
    y_th = (V_th_rel - mu) / sigma
    y_r = (V_0_rel - mu) / sigma
    y_th = np.atleast_1d(y_th)
    y_r = np.atleast_1d(y_r)
    # this brings tau_m and tau_r into the correct vectorized form if they are
    # scalars and doesn't do anything if they are arrays of appropriate size
    tau_m = tau_m + y_th - y_th
    tau_r = tau_r + y_th - y_th
    assert y_th.shape == y_r.shape
    assert y_th.ndim == y_r.ndim == 1
    if np.any(V_th_rel - V_0_rel < 0):
        raise ValueError('V_th should be larger than V_0!')

    # determine order of quadrature
    params = {'start_order': 10, 'epsrel': 1e-12, 'maxiter': 10}
    gl_order = _get_erfcx_integral_gl_order(y_th=y_th, y_r=y_r, **params)

    # separate domains
    mask_exc = y_th < 0
    mask_inh = 0 < y_r
    mask_interm = (y_r <= 0) & (0 <= y_th)

    # calculate rescaled siegert
    nu = np.zeros(shape=y_th.shape)
    params = {'tau_m': tau_m[mask_exc], 't_ref': tau_r[mask_exc],
              'gl_order': gl_order}
    nu[mask_exc] = _siegert_exc(y_th=y_th[mask_exc],
                                y_r=y_r[mask_exc], **params)
    params = {'tau_m': tau_m[mask_inh], 't_ref': tau_r[mask_inh],
              'gl_order': gl_order}
    nu[mask_inh] = _siegert_inh(y_th=y_th[mask_inh],
                                y_r=y_r[mask_inh], **params)
    params = {'tau_m': tau_m[mask_interm], 't_ref': tau_r[mask_interm],
              'gl_order': gl_order}
    nu[mask_interm] = _siegert_interm(y_th=y_th[mask_interm],
                                      y_r=y_r[mask_interm], **params)

    # include exponential contributions
    nu[mask_inh] *= np.exp(-y_th[mask_inh]**2)
    nu[mask_interm] *= np.exp(-y_th[mask_interm]**2)

    # convert back to scalar if only one value calculated
    if nu.shape == (1,):
        return nu.item(0)
    else:
        return nu


def _get_erfcx_integral_gl_order(y_th, y_r, start_order, epsrel, maxiter):
    """Determine order of Gauss-Legendre quadrature for erfcx integral."""
    # determine maximal integration range
    a = min(np.abs(y_th).min(), np.abs(y_r).min())
    b = max(np.abs(y_th).max(), np.abs(y_r).max())

    # adaptive quadrature from scipy.integrate for comparison
    I_quad = _quad(_erfcx, a, b, epsabs=0, epsrel=epsrel)[0]

    # increase order to reach desired accuracy
    order = start_order
    for _ in range(maxiter):
        I_gl = _erfcx_integral(a, b, order=order)[0]
        rel_error = np.abs(I_gl / I_quad - 1)
        if rel_error < epsrel:
            return order
        else:
            order *= 2
    msg = f'Quadrature search failed to converge after {maxiter} iterations. '
    msg += f'Last relative error {rel_error:e}, desired {epsrel:e}.'
    raise RuntimeError(msg)


def _erfcx_integral(a, b, order):
    """Fixed order Gauss-Legendre quadrature of erfcx from a to b."""
    assert np.all(a >= 0) and np.all(b >= 0)
    x, w = _roots_legendre(order)
    x = x[:, np.newaxis]
    w = w[:, np.newaxis]
    return (b - a) * np.sum(w * _erfcx((b - a) * x / 2 + (b + a) / 2),
                            axis=0) / 2


def _siegert_exc(y_th, y_r, tau_m, t_ref, gl_order):
    """Calculate Siegert for y_th < 0."""
    assert np.all(y_th < 0)
    Int = _erfcx_integral(np.abs(y_th), np.abs(y_r), gl_order)
    return 1 / (t_ref + tau_m * np.sqrt(np.pi) * Int)


def _siegert_inh(y_th, y_r, tau_m, t_ref, gl_order):
    """Calculate Siegert without exp(-y_th**2) factor for 0 < y_th."""
    assert np.all(0 < y_r)
    e_V_th_2 = np.exp(-y_th**2)
    Int = (2 * _dawsn(y_th) - 2
           * np.exp(y_r**2 - y_th**2) * _dawsn(y_r))
    Int -= e_V_th_2 * _erfcx_integral(y_r, y_th, gl_order)
    return 1 / (e_V_th_2 * t_ref + tau_m * np.sqrt(np.pi) * Int)


def _siegert_interm(y_th, y_r, tau_m, t_ref, gl_order):
    """Calculate Siegert without exp(-y_th**2) factor for y_r <= 0 <= y_th."""
    assert np.all((y_r <= 0) & (0 <= y_th))
    e_V_th_2 = np.exp(-y_th**2)
    Int = 2 * _dawsn(y_th)
    Int += e_V_th_2 * _erfcx_integral(y_th, np.abs(y_r), gl_order)
    return 1 / (e_V_th_2 * t_ref + tau_m * np.sqrt(np.pi) * Int)


def mean_input(network):
    '''
    Calc mean inputs to populations as function of firing rates of populations.

    See :func:`nnmt.lif._general._mean_input` for full documentation.

    Parameters
    ----------
    network : Network object
        Model with the network parameters and previously calculated results
        listed in :func:`nnmt.lif.delta._mean_input`.

    Returns
    -------
    np.array
        Array of mean inputs to each population in V.
    '''
    list_of_params = ['J', 'K', 'tau_m', 'J_ext', 'K_ext', 'nu_ext']
    try:
        params = {key: network.network_params[key] for key in list_of_params}
    except KeyError as param:
        raise RuntimeError(f'You are missing {param} for this calculation.')

    try:
        params['nu'] = network.results[_prefix + 'firing_rates']
    except KeyError as quantity:
        raise RuntimeError(f'You first need to calculate the {quantity}.')

    return _cache(network, _mean_input, params, _prefix + 'mean_input', 'volt')


@_check_positive_params
def _mean_input(nu, J, K, tau_m, J_ext, K_ext, nu_ext):
    """
    Calc mean input for lif neurons in fixed in-degree connectivity network.

    See :func:`nnmt.lif._general._mean_input` for full documentation.

    Parameters
    ----------
    nu : np.array
        Firing rates of populations in Hz.
    J : np.array
        Weight matrix in V.
    K : np.array
        In-degree matrix.
    tau_m : [float | 1d array]
        Membrane time constant of post-synatic neuron in s.
    J_ext : np.array
        External weight matrix in V.
    K_ext : np.array
        Numbers of external input neurons to each population.
    nu_ext : 1d array
        Firing rates of external populations in Hz.

    Returns
    -------
    np.array
        Array of mean inputs to each population in V.
    """
    return _general._mean_input(nu, J, K, tau_m,
                                J_ext, K_ext, nu_ext)


def std_input(network):
    '''
    Calculates standard deviation of inputs to populations.

    See :func:`nnmt.lif._general._std_input` for full documentation.

    Parameters
    ----------
    network : nnmt.models.Network or child class instance.
        Network with the network parameters and previously calculated results
        listed in :func:`nnmt.lif.delta._std_input`.

    Returns
    -------
    np.array
        Array of mean inputs to each population in V.
    '''
    list_of_params = ['J', 'K', 'tau_m', 'J_ext', 'K_ext', 'nu_ext']
    try:
        params = {key: network.network_params[key] for key in list_of_params}
    except KeyError as param:
        raise RuntimeError(f'You are missing {param} for this calculation.')

    try:
        params['nu'] = network.results[_prefix + 'firing_rates']
    except KeyError as quantity:
        raise RuntimeError(f'You first need to calculate the {quantity}.')

    return _cache(network, _std_input, params, _prefix + 'std_input', 'volt')


@_check_positive_params
def _std_input(nu, J, K, tau_m, J_ext, K_ext, nu_ext):
    """
    Plain calculation of standard deviation of neuronal input.

    See :func:`nnmt.lif._general._std_input` for full documentation.

    Parameters
    ----------
    nu : np.array
        Firing rates of populations in Hz.
    J : np.array
        Weight matrix in V.
    K : np.array
        In-degree matrix.
    tau_m : [float | 1d array]
        Membrane time constant of post-synatic neuron in s.
    J_ext : np.array
        External weight matrix in V.
    K_ext : np.array
        Numbers of external input neurons to each population.
    nu_ext : 1d array
        Firing rates of external populations in Hz.

    Returns
    -------
    np.array
        Array of mean inputs to each population in V.
    """
    return _general._std_input(nu, J, K, tau_m,
                               J_ext, K_ext, nu_ext)


def _derivative_of_firing_rates_wrt_mean_input(mu, sigma, V_0_rel, V_th_rel,
                                               tau_m, tau_r):
    """
    Derivative of the stationary firing rate with respect to the mean input.

    See Appendix B in :cite:t:`schuecker2014`.

    Parameters
    ----------
    mu : float
        Mean neuron activity in V.
    sigma : float
        Standard deviation of neuron activity in V.
    V_0_rel : float
        Relative reset potential in V.
    V_th_rel : float
        Relative threshold potential in V.
    tau_m : float
        Membrane time constant of post-synatic neuron in s.
    tau_r : float
        Refractory time in s.

    Returns
    -------
    float
        Zero frequency limit of white noise transfer function in Hz/V.
    """
    if np.any(sigma == 0):
        raise ZeroDivisionError('Phi_prime_mu contains division by sigma!')

    y_th = (V_th_rel - mu) / sigma
    y_r = (V_0_rel - mu) / sigma
    nu0 = _firing_rates_for_given_input(mu, sigma, V_0_rel, V_th_rel, tau_m,
                                        tau_r)
    return (np.sqrt(np.pi) * tau_m * np.power(nu0, 2) / sigma
            * (np.exp(y_th**2) * (1 + _erf(y_th)) - np.exp(y_r**2)
               * (1 + _erf(y_r))))
