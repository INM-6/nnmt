"""
Collection of functions for LIF neurons with exponential synapses.

Network Functions
*****************

.. autosummary::
    :toctree: _toctree/lif/

    firing_rates
    mean_input
    std_input
    working_point
    transfer_function
    fit_transfer_function
    effective_connectivity
    propagator
    sensitivity_measure
    sensitivity_measure_all_eigenmodes
    power_spectra
    external_rates_for_fixed_input

Parameter Functions
*******************

.. autosummary::
    :toctree: _toctree/lif/

    _firing_rates
    _firing_rate_shift
    _firing_rate_taylor
    _mean_input
    _std_input
    _transfer_function
    _transfer_function_shift
    _transfer_function_taylor
    _fit_transfer_function
    _derivative_of_firing_rates_wrt_mean_input
    _derivative_of_firing_rates_wrt_input_rate
    _effective_connectivity
    _propagator
    _match_eigenvalues_across_frequencies
    _sensitivity_measure
    _sensitivity_measure_all_eigenmodes
    _power_spectra
    _external_rates_for_fixed_input

"""

import warnings
from collections import defaultdict
import numpy as np
import mpmath
import scipy.linalg as slinalg
from scipy.special import (
    erf as _erf,
    zetac as _zetac,
    erfcx as _erfcx,
    )

from ..utils import (_check_positive_params,
                     _check_k_in_fast_synaptic_regime,
                     _cache)

from . import _general

from .delta import (
    _firing_rates_for_given_input as _delta_firing_rate,
    _derivative_of_firing_rates_wrt_mean_input
    as _derivative_of_delta_firing_rates_wrt_mean_input,
    _get_erfcx_integral_gl_order,
    _siegert_exc,
    _siegert_inh,
    _siegert_interm,
    )

pcfu_vec = np.frompyfunc(mpmath.pcfu, 2, 1)

_prefix = 'lif.exp.'


def working_point(network, method='shift', **kwargs):
    """
    Calculates working point for exp PSCs.

    Parameters
    ----------
    network : nnmt.models.Network or child class instance.
        Network with the network parameters listed in
        :func:`nnmt.lif.exp._firing_rates`.
    method : {'shift', 'taylor'}, optional
        Method used to integrate the adapted Siegert function. Default is
        'shift'.
    kwargs
        For additional kwargs regarding the fixpoint iteration procedure see
        :func:`nnmt.lif._general._firing_rate_integration`.

    Returns
    -------
    dict
        Dictionary containing firing rates, mean input and std input.
    """
    return {'firing_rates': firing_rates(network, method, **kwargs),
            'mean_input': mean_input(network),
            'std_input': std_input(network)}


def firing_rates(network, method='shift', **kwargs):
    """
    Calculates stationary firing rates for exp PSCs.

    See :func:`nnmt.lif.exp._firing_rates` for full documentation.

    Parameters
    ----------
    network : nnmt.models.Network or child class instance.
        Network with the network parameters listed in
        :func:`nnmt.lif.exp._firing_rates`.
    method : {'shift', 'taylor'}, optional
        Method used to integrate the adapted Siegert function. Default is
        'shift'.
    kwargs
        For additional kwargs regarding the fixpoint iteration procedure see
        :func:`nnmt.lif._general._firing_rate_integration`.

    Returns
    -------
    np.array
        Array of firing rates of each population in Hz.
    """
    list_of_params = [
        'J', 'K',
        'V_0_rel', 'V_th_rel',
        'tau_m', 'tau_s', 'tau_r',
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

    params['method'] = method
    params.update(kwargs)

    return _cache(network,
                  _firing_rates, params, _prefix + 'firing_rates', 'hertz')


def _firing_rates(J, K, V_0_rel, V_th_rel, tau_m, tau_r, tau_s, J_ext, K_ext,
                  nu_ext, method='shift', **kwargs):
    """
    Calculates stationary firing rates for exp PSCs.

    Calculates the stationary firing rate of a neuron with synaptic filter of
    time constant tau_s driven by Gaussian noise with mean mu and standard
    deviation sigma based on :cite:t:`fourcaud2002`, using either a shift of
    the integration boundaries in the white noise Siegert formula, as derived
    in :cite:t:`schuecker2015` (default), or a Taylor expansion around
    :math:`k = \sqrt{\\tau_\mathrm{s}/\\tau_\mathrm{m}}` of Eq. 4.33 in
    :cite:t:`fourcaud2002`.

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
        Membrane time constant in s.
    tau_r : [float | 1d array]
        Refractory time in s.
    tau_s : [float | 1d array]
        Synaptic time constant in s.
    J_ext : np.array
        External weight matrix in V.
    K_ext : np.array
        Numbers of external input neurons to each population.
    nu_ext : 1d array
        Firing rates of external populations in Hz.
    method : {'shift', 'taylor'}, optional
        Method used to integrate the adapted Siegert function. Default is
        'shift'.
    kwargs
        For additional kwargs regarding the fixpoint iteration procedure see
        :func:`nnmt.lif._general._firing_rate_integration`.

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
        'tau_s': tau_s,
        }
    input_params = {
        'J': J,
        'K': K,
        'tau_m': tau_m,
        'J_ext': J_ext,
        'K_ext': K_ext,
        'nu_ext': nu_ext,
        }

    if method == 'shift':
        return _general._firing_rate_integration(_firing_rate_shift,
                                                 firing_rate_params,
                                                 input_params, **kwargs)
    elif method == 'taylor':
        return _general._firing_rate_integration(_firing_rate_taylor,
                                                 firing_rate_params,
                                                 input_params, **kwargs)


@_check_positive_params
@_check_k_in_fast_synaptic_regime
def _firing_rate_shift(V_0_rel, V_th_rel, mu, sigma, tau_m, tau_r, tau_s):
    """
    Calculates stationary firing rates including synaptic filtering.

    Based on :cite:t:`fourcaud2002`, using shift of the integration boundaries
    in the white noise Siegert formula, as derived in :cite:t:`schuecker2015`.

    Parameters
    ----------
    V_th_rel : [float | np.array]
        Relative threshold potential in V.
    V_0_rel : [float | np.array]
        Relative reset potential in V.
    mu : [float | np.array]
        Mean neuron activity in V.
    sigma : [float | np.array]
        Standard deviation of neuron activity in V.
    tau_m : [float | 1d array]
        Membrane time constant in s.
    tau_r : [float | 1d array]
        Refractory time in s.
    tau_s : [float | 1d array]
        Synaptic time constant in s.

    Returns
    -------
    [float | np.array]
        Stationary firing rate in Hz.
    """
    alpha = __alpha_voltage_shift()
    # effective threshold
    # additional factor sigma is canceled in siegert
    V_th1 = V_th_rel + sigma * alpha / 2. * np.sqrt(tau_s / tau_m)
    # effective reset
    V_01 = V_0_rel + sigma * alpha / 2. * np.sqrt(tau_s / tau_m)
    # use standard Siegert with modified threshold and reset
    return _delta_firing_rate(V_01, V_th1, mu, sigma, tau_m, tau_r)


@_check_positive_params
@_check_k_in_fast_synaptic_regime
def _firing_rate_taylor(V_0_rel, V_th_rel, mu, sigma, tau_m, tau_r, tau_s):
    """
    Calcs stationary firing rates for exp PSCs using a Taylor expansion.

    Calculates the stationary firing rate of a neuron with synaptic filter of
    time constant tau_s driven by Gaussian noise with mean mu and standard
    deviation sigma, using Eq. 4.33 in :cite:t:`fourcaud2002` with Taylor
    expansion around :math:`k = \sqrt{\\tau_\mathrm{s}/\\tau_\mathrm{m}}`.

    Parameters
    ----------
    V_th_rel : [float | np.array]
        Relative threshold potential in V.
    V_0_rel : [float | np.array]
        Relative reset potential in V.
    mu : [float | np.array]
        Mean neuron activity in V.
    sigma : [float | np.array]
        Standard deviation of neuron activity in V.
    tau_m : [float | 1d array]
        Membrane time constant in s.
    tau_r : [float | 1d array]
        Refractory time in s.
    tau_s : [float | 1d array]
        Synaptic time constant in s.

    Returns
    -------
    [float | np.array]
        Stationary firing rate in Hz.
    """
    alpha = __alpha_voltage_shift()

    nu0 = _delta_firing_rate(V_0_rel, V_th_rel, mu, sigma, tau_m, tau_r)
    nu0_dPhi = _nu0_dPhi(V_0_rel, V_th_rel, mu, sigma, tau_m, tau_r)
    result = nu0 * (1 - np.sqrt(tau_s * tau_m / 2) * alpha * nu0_dPhi)
    if np.any(result < 0):
        warnings.warn("Negative firing rates detected. You might be in an "
                      "invalid regime. Use `method='shift'` for "
                      "calculating the firing rates instead.")

    if result.shape == (1,):
        return result.item(0)
    else:
        return result


def __alpha_voltage_shift():
    r"""
    Returns the constant :math:`\sqrt{2}|\zeta(1/2)|`.

    Uses zetac, which returns zeta - 1, because zeta is returning nan for
    arguments smaller 1.
    """
    return np.sqrt(2) * abs(_zetac(0.5) + 1)


def _nu0_dPhi(V_0_rel, V_th_rel, mu, sigma, tau_m, tau_r):
    """Calculate nu0 * ( Phi(sqrt(2)*y_th) - Psi(sqrt(2)*y_r) ) safely."""
    # bring into appropriate shape
    V_0_rel, V_th_rel, mu, sigma, tau_m, tau_r = _equalize_shape(
        V_0_rel, V_th_rel, mu, sigma, tau_m, tau_r)

    y_th = (V_th_rel - mu) / sigma
    y_r = (V_0_rel - mu) / sigma

    assert y_th.shape == y_r.shape
    assert y_th.ndim == y_r.ndim == 1

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

    # calculate rescaled Phi
    Phi_th = np.zeros(shape=y_th.shape)
    Phi_r = np.zeros(shape=y_r.shape)
    Phi_th[mask_exc] = _Phi_neg(s=np.sqrt(2) * y_th[mask_exc])
    Phi_r[mask_exc] = _Phi_neg(s=np.sqrt(2) * y_r[mask_exc])
    Phi_th[mask_inh] = _Phi_pos(s=np.sqrt(2) * y_th[mask_inh])
    Phi_r[mask_inh] = _Phi_pos(s=np.sqrt(2) * y_r[mask_inh])
    Phi_th[mask_interm] = _Phi_pos(s=np.sqrt(2) * y_th[mask_interm])
    Phi_r[mask_interm] = _Phi_neg(s=np.sqrt(2) * y_r[mask_interm])

    # include exponential contributions
    Phi_r[mask_inh] *= np.exp(-y_th[mask_inh]**2 + y_r[mask_inh]**2)
    Phi_r[mask_interm] *= np.exp(-y_th[mask_interm]**2)

    # calculate nu * dPhi
    nu_dPhi = nu * (Phi_th - Phi_r)

    # convert back to scalar if only one value calculated
    if nu_dPhi.shape == (1,):
        return nu_dPhi.item(0)
    else:
        return nu_dPhi


def _Phi_neg(s):
    """Calculate Phi(s) for negative arguments"""
    assert np.all(s <= 0)
    return np.sqrt(np.pi / 2.) * _erfcx(np.abs(s) / np.sqrt(2))


def _Phi_pos(s):
    """Calculate Phi(s) without exp(-s**2 / 2) factor for positive arguments"""
    assert np.all(s >= 0)
    return np.sqrt(np.pi / 2.) * (2 - np.exp(-s**2 / 2.)
                                  * _erfcx(s / np.sqrt(2)))


def mean_input(network):
    '''
    Calc mean inputs to populations as function of firing rates of populations.

    Following :cite:t:`fourcaud2002`.

    Parameters
    ----------
    network : nnmt.models.Network or child class instance.
        Network with the network parameters and previously calculated results
        listed in :func:`nnmt.lif.exp._mean_input`.

    Network results
    ---------------
    nu : Quantity(np.array, 'hertz')
        Firing rates of populations in Hz.

    Network parameters
    ------------------
    J : np.array
        Weight matrix in V.
    K : np.array
        Indegree matrix.
    tau_m : [float | 1d array]
        Membrane time constant in s.
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
    '''
    list_of_params = ['J', 'K', 'tau_m', 'J_ext', 'K_ext', 'nu_ext']
    try:
        params = {key: network.network_params[key] for key in list_of_params}
    except KeyError as param:
        raise RuntimeError(f'You are missing {param} for this calculation.')

    try:
        params['nu'] = network.results['lif.exp.firing_rates']
    except KeyError as quantity:
        raise RuntimeError(f'You first need to calculate the {quantity}.')

    return _cache(network, _mean_input, params, _prefix + 'mean_input', 'volt')


@_check_positive_params
def _mean_input(nu, J, K, tau_m, J_ext, K_ext, nu_ext):
    """
    Calc mean input for lif neurons in fixed in-degree connectivity network.

    Parameters
    ----------
    nu : np.array
        Firing rates of populations in Hz.
    J : np.array
        Weight matrix in V.
    K : np.array
        In-degree matrix.
    tau_m : [float | 1d array]
        Membrane time constant in s.
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

    Following :cite:t:`fourcaud2002`.

    Parameters
    ----------
    network: nnmt.models.Network or child class instance.
        Network with the network parameters and previously calculated results
        listed in :func:`nnmt.lif.exp._std_input`.

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
        params['nu'] = network.results['lif.exp.firing_rates']
    except KeyError as quantity:
        raise RuntimeError(f'You first need to calculate the {quantity}.')

    return _cache(network, _std_input, params, _prefix + 'std_input', 'volt')


@_check_positive_params
def _std_input(nu, J, K, tau_m, J_ext, K_ext, nu_ext):
    """
    Plain calculation of standard deviation of neuronal input.

    Parameters
    ----------
    nu : np.array
        Firing rates of populations in Hz.
    J : np.array
        Weight matrix in V.
    K : np.array
        In-degree matrix.
    tau_m : [float | 1d array]
        Membrane time constant in s.
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


def transfer_function(network, freqs=None, method='shift',
                      synaptic_filter=True):
    """
    Calculates the transfer function for each population for given frequencies.

    Requires the computation of :func:`nnmt.lif.exp.mean_input` and
    :func:`nnmt.lif.exp.std_input` first.

    See :func:`nnmt.lif.exp._transfer_function` for full documentation.

    Parameters
    ----------
    network : nnmt.create.Network or child class instance.
        Network with the parameters listed in
        :func:`nnmt.lif.exp._transfer_function`.
    freqs : np.ndarray
        Frequencies for which transfer function should be calculated. You can
        use this if you do not want to use the networks analysis_params.
    method : {'shift', 'taylor'}
        Method used to calculate the transfer function. Default is 'shift'.
    synaptic_filter : bool
        Whether an additional synaptic low pass filter is to be used or not.
        Default is True.

    Returns
    -------
    ureg.Quantity(np.array, 'hertz/millivolt'):
        Transfer functions for each population with the following shape:
        (number of frequencies, number of populations)
    """

    list_of_params = ['tau_m', 'tau_s', 'tau_r', 'V_th_rel', 'V_0_rel']

    try:
        params = {key: network.network_params[key] for key in list_of_params}
        if freqs is None:
            params['omegas'] = network.analysis_params['omegas']
        else:
            params['omegas'] = freqs * 2 * np.pi
    except KeyError as param:
        raise RuntimeError(
            f"You are missing {param} for calculating the transfer function!\n"
            "Have a look into the documentation for more details on 'lif' "
            "parameters.")
    try:
        params['mu'] = network.results['lif.exp.mean_input']
        params['sigma'] = network.results['lif.exp.std_input']
    except KeyError as quantity:
        raise RuntimeError(f'You first need to calculate the {quantity}.')

    params['synaptic_filter'] = synaptic_filter

    if method == 'shift':
        return _cache(network, _transfer_function_shift, params,
                      _prefix + 'transfer_function',
                      'hertz / volt')
    elif method == 'taylor':
        return _cache(network, _transfer_function_taylor, params,
                      _prefix + 'transfer_function',
                      'hertz / volt')


def _transfer_function(mu, sigma, tau_m, tau_s, tau_r, V_th_rel, V_0_rel,
                       omegas, method='shift', synaptic_filter=True):
    """
    Calculates the transfer function at given angular frequencies ``omegas``.

    Either :func:`nnmt.lif.exp._transfer_function_shift` (default) or
    :func:`nnmt.lif.exp._transfer_function_taylor` is used.

    Parameters
    ----------
    mu : [float | np.array]
        Mean neuron activity of one population in V.
    sigma : [float | np.array]
        Standard deviation of neuron activity of one population in V.
    tau_m : [float | np.array]
        Membrane time constant in s.
    tau_s : [float | np.array]
        Synaptic time constant in s.
    tau_r : [float | np.array]
        Refractory time in s.
    V_th_rel : [float | np.array]
        Relative threshold potential in V.
    V_0_rel : [float | np.array]
        Relative reset potential in V.
    omegas : [float | np.array]
        Input angular frequency to population in Hz.
    method : {'shift', 'taylor'}, optional
        Method used to integrate the adapted Siegert function. Default is
        'shift'.
    synaptic_filter : bool
        Whether an additional synaptic low pass filter is to be used or not.
        Default is True.

    Returns
    -------
    [float | np.array]
        Transfer function in Hz/V.
    """

    if method == 'shift':
        return _transfer_function_shift(mu, sigma, tau_m, tau_s, tau_r,
                                        V_th_rel, V_0_rel, omegas,
                                        synaptic_filter)
    elif method == 'taylor':
        return _transfer_function_taylor(mu, sigma, tau_m, tau_s, tau_r,
                                         V_th_rel, V_0_rel, omegas,
                                         synaptic_filter)


@_check_positive_params
@_check_k_in_fast_synaptic_regime
def _transfer_function_shift(mu, sigma, tau_m, tau_s, tau_r, V_th_rel,
                             V_0_rel, omegas, synaptic_filter=True):
    """
    Calcs value of transfer func for one population at given frequency omega.

    Calculates transfer function according to :math:`\tilde{n}` in
    :cite:t:`schuecker2015`. The expression is to first order equivalent to
    :func:`nnmt.lif.exp._transfer_function_taylor`. Since the underlying theory
    is correct to first order, the two expressions are exchangeable.

    The difference here is that the linear response of the system is considered
    with respect to a perturbation of the input to the current I, leading to an
    additional low pass filtering 1/(1+i omega tau_s). Compare with the second
    equation of Eq. 18 and the text below Eq. 29.

    Parameters
    ----------
    mu : [float | np.array]
        Mean neuron activity of one population in V.
    sigma : [float | np.array]
        Standard deviation of neuron activity of one population in V.
    tau_m : [float | np.array]
        Membrane time constant in s.
    tau_s : [float | np.array]
        Synaptic time constant in s.
    tau_r : [float | np.array]
        Refractory time in s.
    V_th_rel : [float | np.array]
        Relative threshold potential in V.
    V_0_rel : [float | np.array]
        Relative reset potential in V.
    omegas : [float | np.array]
        Input angular frequency to population in Hz.
    synaptic_filter : bool
        Whether an additional synaptic low pass filter is to be used or not.
        Default is True.

    Returns
    -------
    [float | np.array]
        Transfer function in Hz/V.
    """
    # ensure right vectorized format
    omegas = np.atleast_1d(omegas)
    mu, sigma, tau_m, tau_r, tau_s, V_th_rel, V_0_rel = (
        _equalize_shape(mu, sigma, tau_m, tau_r, tau_s, V_th_rel, V_0_rel))

    # effective threshold and reset
    alpha = __alpha_voltage_shift()
    V_th_shifted = V_th_rel + sigma * alpha / 2. * np.sqrt(tau_s / tau_m)
    V_0_shifted = V_0_rel + sigma * alpha / 2. * np.sqrt(tau_s / tau_m)

    zero_omega_mask = np.abs(omegas) < 1e-15
    regular_mask = np.invert(zero_omega_mask)

    result = np.zeros((len(omegas), len(mu)), dtype=complex)

    # for frequency zero the exact expression is given by the derivative of
    # f-I-curve
    if np.any(zero_omega_mask):
        result[zero_omega_mask] = (
            _derivative_of_delta_firing_rates_wrt_mean_input(
                V_0_shifted, V_th_shifted, mu, sigma, tau_m, tau_r))

    if np.any(regular_mask):
        nu = _delta_firing_rate(V_0_shifted, V_th_shifted, mu, sigma, tau_m,
                                tau_r)
        nu = np.atleast_1d(nu)[np.newaxis]
        x_t = np.sqrt(2.) * (V_th_shifted - mu) / sigma
        x_r = np.sqrt(2.) * (V_0_shifted - mu) / sigma
        z = -0.5 + 1j * np.outer(omegas[regular_mask], tau_m)

        frac = ((_d_Psi(z, x_t) - _d_Psi(z, x_r))
                / (_Psi(z, x_t) - _Psi(z, x_r)))

        result[regular_mask] = (np.sqrt(2.)
                                / sigma[np.newaxis] * nu
                                / (1. + 1j * np.outer(omegas[regular_mask],
                                                      tau_m))
                                * frac)
    if synaptic_filter:
        result *= _synaptic_filter(omegas, tau_s)
    return result


@_check_positive_params
@_check_k_in_fast_synaptic_regime
def _transfer_function_taylor(mu, sigma, tau_m, tau_s, tau_r, V_th_rel,
                              V_0_rel, omegas, synaptic_filter=True):
    """
    Calcs value of transfer func for one population at given frequency omega.

    The calculation is done according to Eq. 93 in :cite:t:`schuecker2014`.

    The difference here is that the linear response of the system is considered
    with respect to a perturbation of the input to the current I, leading to an
    additional low pass filtering 1/(1+i omega tau_s).
    Compare with the second equation of Eq. 18 and the text below Eq. 29.

    Parameters
    ----------
    mu : [float | np.array]
        Mean neuron activity of one population in V.
    sigma : [float | np.array]
        Standard deviation of neuron activity of one population in V.
    tau_m : [float | np.array]
        Membrane time constant in s.
    tau_s : [float | np.array]
        Synaptic time constant in s.
    tau_r : [float | np.array]
        Refractory time in s.
    V_th_rel : [float | np.array]
        Relative threshold potential in V.
    V_0_rel : [float | np.array]
        Relative reset potential in V.
    omegas : [float | np.array]
        Input angular frequency to population in Hz.
    synaptic_filter : bool
        Whether an additional synaptic low pass filter is to be used or not.
        Default is True.

    Returns
    -------
    [float | np.array]
        Transfer function in Hz/V.
    """
    # ensure right vectorized format
    omegas = np.atleast_1d(omegas)
    mu, sigma, tau_m, tau_r, tau_s, V_th_rel, V_0_rel = (
        _equalize_shape(mu, sigma, tau_m, tau_r, tau_s, V_th_rel, V_0_rel))

    zero_omega_mask = omegas < 1e-15
    regular_mask = np.invert(zero_omega_mask)

    result = np.zeros((len(omegas), len(mu)), dtype=complex)

    # for frequency zero the exact expression is given by the derivative of
    # f-I-curve
    if np.any(zero_omega_mask):
        result[zero_omega_mask] = (
            _derivative_of_firing_rates_wrt_mean_input(
                V_0_rel, V_th_rel, mu, sigma, tau_m, tau_r, tau_s)
            )

    if np.any(regular_mask):
        delta_rates = _delta_firing_rate(
            V_0_rel, V_th_rel, mu, sigma, tau_m, tau_r)
        delta_rates = np.atleast_1d(delta_rates)[np.newaxis]
        exp_rates = _firing_rate_taylor(
            V_0_rel, V_th_rel, mu, sigma, tau_m, tau_r, tau_s)
        exp_rates = np.atleast_1d(exp_rates)[np.newaxis]

        # effective threshold and reset
        x_t = np.sqrt(2.) * (V_th_rel - mu) / sigma
        x_r = np.sqrt(2.) * (V_0_rel - mu) / sigma

        z = -0.5 + 1j * np.outer(omegas[regular_mask], tau_m)
        alpha = __alpha_voltage_shift()
        k = np.sqrt(tau_s / tau_m)
        A = alpha * tau_m * delta_rates * k / np.sqrt(2)
        a0 = _Psi(z, x_t) - _Psi(z, x_r)
        a1 = (_d_Psi(z, x_t) - _d_Psi(z, x_r)) / a0
        a3 = (A / tau_m / exp_rates
              * (-a1**2 + (_d_2_Psi(z, x_t) - _d_2_Psi(z, x_r)) / a0))
        result[regular_mask] = (
            np.sqrt(2.) / sigma * exp_rates
            / (1. + 1j * np.outer(omegas[regular_mask], tau_m))
            * (a1 + a3))

    if synaptic_filter:
        result *= _synaptic_filter(omegas, tau_s)
    return result


def fit_transfer_function(network):
    """
    Fits the transfer function (tf) of a low-pass filter to the passed tf.

    See :func:`nnmt.lif.exp._fit_transfer_function` for full documentation.

    Parameters
    ----------
    network : nnmt.models.Network or child class instance.
        Network with the network parameters, analysis parameters and results
        listed in :func:`nnmt.lif.exp._fit_transfer_function`.

    Returns
    -------
    transfer_function_fit : np.array
        Fit of transfer functions in Hertz/volt for each population with the
        following shape: (number of freqencies, number of populations).
    tau_rate : np.array
        Fitted time constant for each population in s.
    W_rate : np.array
        Matrix of fitted weights (unitless).
    fit_error : float
        Combined fit error.
    """
    list_of_params = ['tau_m', 'J', 'K']

    try:
        params = {key: network.network_params[key] for key in list_of_params}
        params['omegas'] = network.analysis_params['omegas']
    except KeyError as param:
        raise RuntimeError(
            f"You are missing {param} for fitting the transfer function!\n"
            "Have a look into the documentation for more details on 'lif' "
            "parameters.")
    try:
        params['transfer_function'] = (
            network.results['lif.exp.transfer_function'])
    except KeyError as quantity:
        raise RuntimeError(f'You first need to calculate the {quantity}.')

    return _cache(network, _fit_transfer_function, params,
                  [_prefix + 'transfer_function_fit',
                   _prefix + 'tau_rate',
                   _prefix + 'W_rate',
                   _prefix + 'fit_error'],
                  ['hertz / volt',
                   'seconds',
                   None,
                   None])


@_check_positive_params
def _fit_transfer_function(transfer_function, omegas, tau_m, J, K):
    """
    Fits the transfer function (tf) of a low-pass filter to the passed tf.

    For details of the fitting procedure see
    :func:`nnmt._general._fit_transfer_function`.

    For details of the theory refer to
    :cite:t:`senk2020`, Sec. F 'Comparison of neural-field and spiking models'.

    Parameters
    ----------
    transfer_function : np.array
        Transfer functions for each population with the following shape:
        (number of freqencies, number of populations).
    omegas : [float | np.ndarray]
        Input frequencies to population in Hz.
    tau_m : float
        Membrane time constant in s.
    J : np.array
        Weight matrix in V.
    K : np.array
        Indegree matrix.

    Returns
    -------
    transfer_function_fit : np.array
        Fit of transfer functions in Hertz/volt for each population with the
        following shape: (number of freqencies, number of populations).
    tau_rate : np.array
        Fitted time constant for each population in s.
    W_rate : np.array
        Matrix of fitted weights (unitless).
    fit_error : float
        Combined fit error.
    """
    transfer_function_fit, tau_rate, h0, fit_error = \
        _general._fit_transfer_function(transfer_function, omegas)

    # weight matrix of rate model
    W_rate = h0 * tau_m * J * K

    return transfer_function_fit, tau_rate, W_rate, fit_error


def _synaptic_filter(omegas, tau_s):
    """Additional low-pass filter due to perturbation to the input current."""
    return 1 / (1. + 1j * np.outer(omegas, tau_s))


def _equalize_shape(*args):
    """Brings list of arrays and scalars into similar 1d shape if possible."""
    args = [np.atleast_1d(arg) for arg in args]
    max_arg = args[0]
    for arg in args[1:]:
        if len(arg) > len(max_arg):
            max_arg = arg
    args = [_similar_array(arg, max_arg) for arg in args]
    return args


def _similar_array(x, array):
    """Returns an array of x of similar shape as array."""
    x = np.atleast_1d(x)
    if x.shape == array.shape:
        return x
    elif len(x) == 1:
        return np.ones(array.shape) * x
    else:
        raise RuntimeError(f'Unclear how to shape {x} into shape of {array}.')


@_check_positive_params
@_check_k_in_fast_synaptic_regime
def _derivative_of_firing_rates_wrt_mean_input(V_0_rel, V_th_rel, mu, sigma,
                                               tau_m, tau_r, tau_s):
    """
    Derivative of the stationary firing rates with respect to the mean input.

    See Appendix B in :cite:t:`schuecker2014`.

    Parameters
    ----------
    tau_m : [float | np.ndarray]
        Membrane time constant in s.
    tau_s : [float | np.ndarray]
        Synaptic time constant in s.
    tau_r : [float | np.ndarray]
        Refractory time in s.
    V_th_rel : [float | np.ndarray]
        Relative threshold potential in V.
    V_0_rel : [float | np.ndarray]
        Relative reset potential in V.
    mu : [float | np.ndarray]
        Mean neuron activity in V.
    sigma : [float | np.ndarray]
        Standard deviation of neuron activity in V.

    Returns
    -------
    float
        Zero frequency limit of colored noise transfer function in Hz/V.
    """
    if np.any(sigma == 0):
        raise ZeroDivisionError('Function contains division by sigma!')

    alpha = __alpha_voltage_shift()
    x_th = np.sqrt(2) * (V_th_rel - mu) / sigma
    x_r = np.sqrt(2) * (V_0_rel - mu) / sigma
    integral = 1 / tau_m / _delta_firing_rate(
        V_0_rel, V_th_rel, mu, sigma, tau_m, tau_r)
    prefactor = np.sqrt(tau_s / tau_m) * alpha / (tau_m * np.sqrt(2))
    dnudmu = _derivative_of_delta_firing_rates_wrt_mean_input(
        V_0_rel, V_th_rel, mu, sigma, tau_m, tau_r)
    dPhi_prime = _Phi_prime_mu(x_th, sigma) - _Phi_prime_mu(x_r, sigma)
    dPhi = _Phi(x_th) - _Phi(x_r)
    phi = dPhi_prime * integral + (2 * np.sqrt(2) / sigma) * dPhi**2
    return dnudmu - prefactor * phi / integral**3


def _Phi(s):
    """
    Helper function to calc stationary firing rates with synaptic filtering.

    Corresponds to u^-2 F in Eq. 53 of :cite:t:`schuecker2014`.
    """
    return np.sqrt(np.pi / 2.) * (np.exp(s**2 / 2.)
                                  * (1 + _erf(s / np.sqrt(2))))


def _Psi(z, x):
    """
    Calcs Psi(z,x)=exp(x**2/4)*U(z,x), with U(z,x) the parabolic cylinder func.
    """
    parabolic_cylinder_fn = pcfu_vec(z, -x).astype(complex)
    return np.exp(0.25 * x**2) * parabolic_cylinder_fn


def _d_Psi(z, x):
    """
    First derivative of Psi using recurrence relations.

    (Eq.: 12.8.9 in http://dlmf.nist.gov/12.8)
    """
    return (1. / 2. + z) * _Psi(z + 1, x)


def _d_2_Psi(z, x):
    """
    Second derivative of Psi using recurrence relations.

    (Eq.: 12.8.9 in http://dlmf.nist.gov/12.8)
    """
    return (1. / 2. + z) * (3. / 2. + z) * _Psi(z + 2, x)


def _Phi_prime_mu(s, sigma):
    """
    Derivative of the helper function _Phi(s) with respect to the mean input.
    """
    if np.any(sigma < 0):
        raise ValueError('sigma needs to be larger than zero!')
    if np.any(sigma == 0):
        raise ZeroDivisionError('Function contains division by sigma!')

    return -np.sqrt(np.pi) / sigma * (s * np.exp(s**2 / 2.)
                                      * (1 + _erf(s / np.sqrt(2)))
                                      + np.sqrt(2) / np.sqrt(np.pi))


@_check_positive_params
@_check_k_in_fast_synaptic_regime
def _derivative_of_firing_rates_wrt_input_rate(
        mu, sigma, tau_m, tau_s, tau_r, V_th_rel, V_0_rel, j):
    """
    Derivative of the stationary firing rates with respect to input rate.

    Parameters
    ----------
    mu : [float | np.ndarray]
        Mean neuron activity in V.
    sigma :
        Standard deviation of neuron activity in V.
    tau_m : [float | np.ndarray]
        Membrane time constant in s.
    tau_s : [float | np.ndarray]
        Synaptic time constant in s.
    tau_r : [float | np.ndarray]
        Refractory time in s.
    V_th_rel : [float | np.ndarray]
        Relative threshold potential in V.
    V_0_rel : [float | np.ndarray]
        Relative reset potential in V.
    j : float
        Effective connectivity weight in V.

    Returns
    -------
    float
        Unitless derivative.
    """

    try:
        if any(sigma == 0 for sigma in sigma):
            raise ZeroDivisionError('Phi_prime_mu contains division by sigma!')
    except TypeError:
        if sigma == 0:
            raise ZeroDivisionError('Phi_prime_mu contains division by sigma!')

    alpha = __alpha_voltage_shift()

    y_th = (V_th_rel - mu) / sigma
    y_r = (V_0_rel - mu) / sigma

    y_th_fb = y_th + alpha / 2. * np.sqrt(tau_s / tau_m)
    y_r_fb = y_r + alpha / 2. * np.sqrt(tau_s / tau_m)

    nu0 = _firing_rate_shift(V_0_rel, V_th_rel, mu, sigma, tau_m, tau_r, tau_s)

    # linear contribution
    lin = (np.sqrt(np.pi) * (tau_m * nu0)**2 * j / sigma
           * (np.exp(y_th_fb**2) * (1 + _erf(y_th_fb))
              - np.exp(y_r_fb**2)
              * (1 + _erf(y_r_fb))))

    # quadratic contribution
    sqr = (np.sqrt(np.pi) * (tau_m * nu0)**2 * j / sigma
           * (np.exp(y_th_fb**2) * (1 + _erf(y_th_fb))
              * 0.5 * y_th * j / sigma - np.exp(y_r_fb**2)
              * (1 + _erf(y_r_fb)) * 0.5 * y_r * j / sigma))

    return lin + sqr


def effective_connectivity(network):
    """
    Effective connectivity for different frequencies.

    Note that the frequencies of the transfer function and the delay
    distribution matrix need to be matching.

    Parameters
    ----------
    network: nnmt.models.Network or child class instance.
        Network with the network parameters and previously calculated results
        listed in the following.

    Network results
    ---------------
    transfer_function : np.ndarray
        Transfer function for given frequencies in Hz/V.

    Network Parameters
    ------------------
    D : np.ndarray
        Unitless delay distribution of shape
        (len(omegas), len(populations), len(populations)).
    J : np.ndarray
        Weight matrix in V.
    K : np.ndarray
        Indegree matrix.
    tau_m : float
        Membrane time constant in s.

    Returns:
    --------
    np.ndarray
        Effective connectivity matrix.
    """

    list_of_params = ['J', 'K', 'tau_m']

    try:
        params = {key: network.network_params[key] for key in list_of_params}
    except KeyError as param:
        raise RuntimeError(
            f"You are missing {param} for calculating the effective "
            "connectivity!\n"
            "Have a look into the documentation for more details on 'lif' "
            "parameters.")
    try:
        params['transfer_function'] = (
            network.results['lif.exp.transfer_function'])
        params['D'] = network.results['D']
    except KeyError as quantity:
        raise RuntimeError(f'You first need to calculate the {quantity}.')

    return _cache(network, _effective_connectivity, params,
                  _prefix + 'effective_connectivity')


@_check_positive_params
def _effective_connectivity(transfer_function, D, J, K, tau_m):
    """
    Effective connectivity for different frequencies.

    See equation 12 and following in Bos 2015.

    Note that the frequencies of the transfer function and the delay
    distribution matrix need to be matching.

    Network results
    ---------------
    transfer_function : np.ndarray
        Transfer_function for given frequencies in hertz/V.

    Parameters
    ----------
    D : np.ndarray
        Unitless delay distribution of shape
        (len(omegas), len(populations), len(populations)).
    J : np.ndarray
        Weight matrix in V.
    K : np.ndarray
        Indegree matrix.
    tau_m : float
        Membrane time constant in s.

    Returns
    -------
    np.ndarray
        Effective connectivity matrix.
    """
    # This ensures that it also works if transfer function has only been
    # calculated for a single frequency. But it should be removed once we have
    # made sure that the frequency dependend quantities always return an object
    # with the frequencies indexed by the first axis.
    if len(D.shape) == 1:
        tf = transfer_function
    elif len(D.shape) == 2:
        tf = np.tile(transfer_function, (K.shape[0], 1)).T
    elif len(D.shape) == 3:
        tf = np.tile(transfer_function.T, (K.shape[0], 1, 1))
        tf = np.einsum('ijk->kji', tf)
    else:
        raise RuntimeError('Delay distribution matrix has no valid format.')
    return tau_m * J * K * tf * D


def propagator(network):
    """
    Propagator for different frequencies.

    Parameters
    ----------
    network: nnmt.models.Network or child class instance.
        Network with the network parameters and previously calculated results
        listed in the following.

    Network results
    ---------------
    effective_connectivity : np.ndarray
        Effective connectivity matrix.

    Returns:
    --------
    np.ndarray
        Propagator for different frequencies. Shape:
        (num freqs, num populations, num populations).
    """
    params = {}
    try:
        params['effective_connectivity'] = (
            network.results['lif.exp.effective_connectivity'])
    except KeyError as quantity:
        raise RuntimeError(f'You first need to calculate the {quantity}.')

    return _cache(network, _propagator, params, _prefix + 'propagator')


def _propagator(effective_connectivity):
    """
    Propagator of network.

    Parameters
    ----------
    effective_connectivity : np.ndarray
        Effective connectivity matrix.

    Returns
    -------
    np.ndarray
        Propagator.
    """
    Q = np.linalg.inv(np.identity(effective_connectivity.shape[-1])
                      - effective_connectivity)
    prop = np.array([np.dot(q, e)
                     for (q, e) in zip(Q, effective_connectivity)])
    return prop


def _match_eigenvalues_across_frequencies(eigenvalues, margin=1e-5):
    """
    Resorts the eigenvalues of the effective connectivity matrix.

    The eigenvalues of the effective connectivity are calculated once
    per frequency. To link the eigenvalues/eigenmodes across frequencies this
    utility function calculates the distance between subsequent (in frequency)
    eigenvalues and matches them if the distance is smaller equal the margin.

    This is done to obtain eigenvalue trajectories as in Fig. 4 of
    :cite:t:`bos2016`.

    If just two eigenvalue have a larger distance than the margin, they can
    be immediately swapped. If more eigenvalues have a larger distance, the
    resorting is more complicated (multi-swaps).

    The default value for the margin is chosen from experience. The smaller
    the frequency step in in the analysis frequencies, the smaller the margin
    needs to be chosen. **It is recommended to cross-check the resorting by
    plotting the eigenvalues across frequencies in the complex plane.**

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues of the effective connectivity matrix
        Shape: (num analysis freqs, num populations)
    margin : float
        Maximal allowed distance between the eigenvalues of the effective
        connectivity matrix at two subsequent frequencies.

    Returns
    -------
    np.ndarray
        Resorted eigenvalues.
    np.ndarray
        Mapping from old to new indices (e.g. for resorting the eigenmodes).
        Shape: (num analysis freqs, num populations)
    """
    eig = eigenvalues.copy()

    n_freqs = eig.shape[0]
    n_evals = eig.shape[1]

    # define vector of eigenvalues at frequency 0
    previous = eig[0, :]

    # initialize containers
    distances = np.zeros([n_freqs - 1, n_evals])
    multi_swaps = {}
    resorted_eigenvalues_mask = np.tile(
        np.arange(n_evals), (n_freqs, 1))

    # loop over all frequencies > 0
    for i in range(1, n_freqs):
        # compare new to previous
        new = eig[i, :]
        distances[i-1, :] = abs(previous - new)

        # get all distances which are larger than margin
        if np.any(distances[i-1, :] > margin):
            indices = np.argwhere(distances[i-1, :] > margin).reshape(-1)
            # if more than two or more eigenvalues need to be
            # swapped, store this in multi_swaps dictionary
            if len(indices) >= 2:
                multi_swaps[i-1] = indices

        previous = new

    if multi_swaps:
        # loop over all frequencies with required multi_swaps
        for n, (i, j) in enumerate(zip(list(multi_swaps.keys())[:-1],
                                       list(multi_swaps.keys())[1:])):
            # i is the frequency at index n
            # j corresponds to frequency at index n+1

            # duplicate the eigenvalues again
            original = eig.copy()

            # loop over the eigenvalues indices that need swapping at
            # the frequencies corresponding to index n
            indices_to_swap = list(multi_swaps.values())[n]
            for k in indices_to_swap:
                # determine the minimal distance of this one eigenvalue
                # to the eigenvalues at the next frequency step
                index = np.argmin(
                    np.abs(original[i+1, indices_to_swap] - original[i, k]))

                # resort this one eigenvalue until the next frequency index
                # for which a multi-swap is necessary
                eig[i+1:j+1, k] = original[i+1:j+1, indices_to_swap[index]]
                resorted_eigenvalues_mask[i+1:j+1, k] = indices_to_swap[index]

            # check for ambiguous resorting
            _, counts = np.unique(resorted_eigenvalues_mask[i+1, :],
                                  return_counts=True)
            if any(counts >= 2):
                warnings.warn(
                    'Ambiguous eigenvalue resorting detected at frequency '
                    f'index {i+1}. '
                    'Please cross-check the sorting and consider modifying '
                    'the margin and/or the frequency resolution.')

        # deal with the last swap
        original = eig.copy()
        i = list(multi_swaps.keys())[-1]
        indices_to_swap = list(multi_swaps.values())[-1]
        for k in indices_to_swap:
            index = np.argmin(
                np.abs(original[i+1, indices_to_swap] - original[i, k]))
            eig[i+1, k] = original[i+1, indices_to_swap[index]]
            resorted_eigenvalues_mask[i+1, k] = indices_to_swap[index]

        # check for ambiguous resorting
        _, counts = np.unique(resorted_eigenvalues_mask[i+1, :],
                              return_counts=True)
        if any(counts >= 2):
            warnings.warn(
                'Ambiguous eigenvalue resorting detected at frequency '
                f'index {i+1}. '
                'Please cross-check the sorting and consider modifying '
                'the margin and/or the frequency resolution.')

    return eig, resorted_eigenvalues_mask


def sensitivity_measure(network, frequency,
                        resorted_eigenvalues_mask='None',
                        eigenvalue_index='None'):
    """
    Calculates sensitivity measure as in Eq. 7 in Bos et al. (2015).

    Evaluates the sensitivity measure at a given frequency. By default,
    the effective connectivity is diagonalized and the eigenmode corresponding
    to the eigenvalue that is closest to the complex(1, 0) is chosen.

    Another eigenmode can be specified by the parameter eigenvalue_index.
    The order of the eigenvalues can be specified by the
    resorted_eigenvalues_mask.

    Parameters
    ----------
    network : nnmt.models.Network or child class instance.
        Network with the network parameters and previously calculated results
        listed in the following.
    frequency : np.float
        Frequency at which the sensitivity is evaluated in Hz.
    resorted_eigenvalues_mask : np.ndarray
        Mapping from old to new indices (e.g. for resorting the eigenmodes) as
        obtained from _match_eigenvalues_across_frequencies.
        Shape : (num populations, num analysis freqs)
    eigenvalue_index : int
        Index specifying the eigenvalue and corresponding eigenmode for which
        the sensitivity measure is evaluated.

    Analysis Parameters
    -------------------
    omegas : float or np.ndarray
        Input frequencies to population in Hz.

    Network results
    ---------------
    effective_connectivity : np.ndarray
        Effective connectivity matrix.

    Returns
    -------
    dict
        Dictionary containing the results of the sensitivity analysis.

        critical_frequency : np.float
            Frequency at which the sensitivity is evaluated in Hz.
        critical_frequency_index : int
            Index of critical_frequency in all analysis frequencies.
        critical_eigenvalue : np.complex
            Critical eigenvalue.
        k : np.complex
            Vector point from critical eigenvalue to complex(1,0).
        k_per : np.complex
            Vector perpendiculat to k.
        sensitivity : np.ndarray
            Sensitivity measure.
            Shape : (num analysis freqs, num populations, num populations)
        sensitivity_amp : np.ndarray
            Projection of sensitivity measure that alters amplitude of
            peak in power spectrum.
            Shape : (num analysis freqs, num populations, num populations)
        sensitivity_freq : np.ndarray
            Projection of Sensitivity measure that alters frequency of
            peak power spectrum.
            Shape : (num analysis freqs, num populations, num populations)
    """
    params = {}
    try:
        params['effective_connectivity'] = (
            network.results['lif.exp.effective_connectivity'])
        params['frequency'] = frequency
        params['analysis_frequencies'] = (
            network.analysis_params['omegas'] / 2 / np.pi)
        params['resorted_eigenvalues_mask'] = resorted_eigenvalues_mask
        params['eigenvalue_index'] = eigenvalue_index

    except KeyError as quantity:
        raise RuntimeError(f'You first need to calculate the {quantity}.')

    return _cache(network, _sensitivity_measure, params,
                  _prefix + 'sensitivity_measure')


@_check_positive_params
def _sensitivity_measure(effective_connectivity, frequency,
                         analysis_frequencies, resorted_eigenvalues_mask,
                         eigenvalue_index):
    """
    Calculates sensitivity measure as in Eq. 7 in Bos et al. (2015).

    Parameters
    ----------
    effective_connectivity : np.ndarray
        Effective connectivity matrix.
    frequency : np.float
        Frequency at which the sensitivity is evaluated in Hz.
    analysis_frequencies : np.ndarray
        Analysis frequencies.
    resorted_eigenvalues_mask : np.ndarray
        Mapping from old to new indices (e.g. for resorting the eigenmodes) as
        obtained from _match_eigenvalues_across_frequencies.
        Shape : (num analysis freqs, num populations)
    eigenvalue_index : int
        Index specifying the eigenvalue and corresponding eigenmode for which
        the sensitivity measure is evaluated.

    Returns
    -------
    dict
        Dictionary containing the results of the sensitivity analysis.

        critical_frequency : np.float
            Frequency at which the sensitivity is evaluated in Hz.
        critical_frequency_index : int
            Index of critical_frequency in all analysis frequencies.
        critical_eigenvalue : np.complex
            Critical eigenvalue.
        k : np.complex
            Vector point from critical eigenvalue to complex(1,0).
        k_per : np.complex
            Vector perpendiculat to k.
        sensitivity : np.ndarray
            Sensitivity measure.
            Shape : (num analysis freqs, num populations, num populations)
        sensitivity_amp : np.ndarray
            Projection of sensitivity measure that alters amplitude of
            peak in power spectrum.
            Shape : (num analysis freqs, num populations, num populations)
        sensitivity_freq : np.ndarray
            Projection of Sensitivity measure that alters frequency of
            peak power spectrum.
            Shape : (num analysis freqs, num populations, num populations)
    """
    frequency_index = np.argmin(
            abs(analysis_frequencies-frequency))

    eff_conn_of_omega = effective_connectivity[frequency_index, :, :]

    # for brevity the sensitivity measure is called T in the following
    T = np.zeros(eff_conn_of_omega.shape, dtype=complex)
    e, U_l, U_r = slinalg.eig(eff_conn_of_omega, left=True, right=True)

    # TODO: currently need to catch this for the fixture creation
    if (isinstance(resorted_eigenvalues_mask, str)
            and resorted_eigenvalues_mask == 'None'):
        resorted_eigenvalues_mask = 'None'

    if resorted_eigenvalues_mask is not 'None':
        # apply the resorting
        e = e[resorted_eigenvalues_mask[frequency_index, :]]
        U_l = U_l[:, resorted_eigenvalues_mask[frequency_index, :]]
        U_r = U_r[:, resorted_eigenvalues_mask[frequency_index, :]]

    if eigenvalue_index is 'None':
        # find eigenvalue closest to one
        eigenvalue_index = np.argmin(np.abs(e - 1))

    T = np.outer(U_l[:, eigenvalue_index].conj(), U_r[:, eigenvalue_index])
    T /= np.dot(U_l[:, eigenvalue_index].conj(), U_r[:, eigenvalue_index])
    T *= eff_conn_of_omega

    critical_eigenvalue = e[eigenvalue_index]
    # vector pointing from critical eigenvalue at frequency to complex(1,0)
    # perturbation shifting critical eigenvalue along k
    # brings eigenvalue towards or away from one,
    # resulting in an increased or
    # decreased peak amplitude in the spectrum
    k = np.asarray([1, 0]) - np.asarray([critical_eigenvalue.real,
                                         critical_eigenvalue.imag])
    # normalize k
    k /= np.sqrt(np.dot(k, k))

    # vector perpendicular to k
    # perturbation shifting critical eigenvalue along k_per
    # alters the trajectory such that it passes closest
    # to one at a lower or
    # higher frequency while conserving the height of the peak
    k_per = np.asarray([-k[1], k[0]])
    # normalize k_per
    k_per /= np.sqrt(np.dot(k_per, k_per))

    # projection of sensitivity measure in to direction
    # that alters amplitude
    sensitivity_amp = T.real*k[0] + T.imag*k[1]
    # projection of sensitivity measure in to direction
    # that alters frequency
    sensitivity_freq = T.real*k_per[0] + T.imag*k_per[1]

    sensitivity_measure = {
            'eigenvalue_index': eigenvalue_index,
            'critical_frequency': frequency,
            'critical_frequency_index': frequency_index,
            'critical_eigenvalue': critical_eigenvalue,
            'k': k,
            'k_per': k_per,
            'sensitivity': T,
            'sensitivity_amp': sensitivity_amp,
            'sensitivity_freq': sensitivity_freq}

    return sensitivity_measure


def sensitivity_measure_all_eigenmodes(network, margin=1e-5):
    """
    Calculates the :func:`_sensitivity_measure` for each eigenmode.

    See :func:`_sensitivity_measure_all_eigenmodes` for full documentation.

    Parameters
    ----------
    network : nnmt.models.Network or child class instance.
        Network with the network parameters and previously calculated results
        listed in the following.
    margin : float
        Maximal allowed distance between the eigenvalues of the effective
        connectivity matrix at two subsequent frequencies.

    Returns
    -------
    dict
        Sensitivity measure dictionary.
    """
    params = {}
    try:
        params['effective_connectivity'] = (
            network.results['lif.exp.effective_connectivity'])
        params['analysis_frequencies'] = (
            network.analysis_params['omegas'] / 2 / np.pi)
        params['margin'] = margin
    except KeyError as quantity:
        raise RuntimeError(f'You first need to calculate the {quantity}.')

    return _cache(network, _sensitivity_measure_all_eigenmodes, params,
                  _prefix + 'sensitivity_measure_all_eigenmodes')


def _sensitivity_measure_all_eigenmodes(effective_connectivity,
                                        analysis_frequencies,
                                        margin=1e-5):
    """
    Calculates the :func:`_sensitivity_measure` for each eigenmode.

    Identifies the frequency which is closest to complex(1,0) for each
    eigenvalue trajectory and evaluates the sensitivity measure, as well as its
    projections on the direction that influences the amplitude and the
    direction that influences the frequency are calculated.

    The results are stored in a dictionary with the eigenvalue index as key
    and the calculated quantities as values.

    Parameters
    ----------
    effective_connectivity : np.ndarray
        Effective connectivity matrix.
        Shape : ()
    analysis_frequencies : np.ndarray
        Analysis frequencies in Hz.
    margin : float
        Maximal allowed distance between the eigenvalues of the effective
        connectivity matrix at two subsequent frequencies.

    Returns
    -------
    dict
        Dictionary of dictionaries containing the sensitivity measure results.
        The dictionary keys are the eigenvalue indices.
    """
    eigenvalues = np.linalg.eig(effective_connectivity)[0]
    resorted_eigenvalues, resorted_eigenvalues_mask = (
        _match_eigenvalues_across_frequencies(eigenvalues, margin=margin))

    sensitivity_measure_dictionary = defaultdict(str)

    # identify frequency which is closest to the point complex(1, 0)
    # per eigenvalue trajectory
    for eig_index, eig in enumerate(resorted_eigenvalues.T):
        critical_frequency = analysis_frequencies[np.argmin(abs(eig-1.0))]

        # unfortunately h5py can't save dictionaries with int as keys, thus
        # the eigenvalue index is stored as a string
        sensitivity_measure_dictionary[str(eig_index)] = _sensitivity_measure(
            effective_connectivity,
            frequency=critical_frequency,
            analysis_frequencies=analysis_frequencies,
            resorted_eigenvalues_mask=resorted_eigenvalues_mask,
            eigenvalue_index=eig_index)

    return dict(sensitivity_measure_dictionary)


def power_spectra(network):
    """
    Calcs vector of power spectra for all populations at given frequencies.

    See: Eq. 18 in Bos et al. (2016)
    Shape of output: (len(omegas), len(populations))

    Parameters
    ----------
    network: nnmt.models.Network or child class instance.
        Network with the network parameters and previously calculated results
        listed in the following.

    Network results
    ---------------
    nu : Quantity(np.array, 'hertz')
        Firing rates of populations in Hz.
    effective_connectivity : np.ndarray
        Effective connectivity matrix.

    Network parameters
    ------------------
    J : np.ndarray
        Weight matrix in V.
    K : np.ndarray
        Indegree matrix.
    N : np.ndarray
        Number of neurons in each population.
    tau_m : [float | np.narray]
        Membrane time constant in s.

    Returns
    -------
    np.ndarray
        Power spectrum in Hz. Shape: (len(freqs), len(populations)).
    """

    list_of_params = ['J', 'K', 'N', 'tau_m']

    try:
        params = {key: network.network_params[key] for key in list_of_params}
    except KeyError as param:
        raise RuntimeError(
            f"You are missing {param} for calculating the effective "
            "connectivity!\n"
            "Have a look into the documentation for more details on 'lif' "
            "parameters.")
    try:
        params['nu'] = network.results['lif.exp.firing_rates']
        params['effective_connectivity'] = (
            network.results['lif.exp.effective_connectivity'])
    except KeyError as quantity:
        raise RuntimeError(f'You first need to calculate the {quantity}.')

    return _cache(network, _power_spectra, params,
                  _prefix + 'power_spectra', 'hertz ** 2')


@_check_positive_params
def _power_spectra(nu, effective_connectivity, J, K, N, tau_m):
    """
    Calcs vector of power spectra for all populations at given frequencies.

    See: Eq. 18 in Bos et al. (2016)
    Shape of output: (len(omegas), len(populations))

    Parameters
    ----------
    nu : np.ndarray
        Firing rates of the different populations in Hz.
    effective_connectivity : np.ndarray
        Effective connectivity matrix.
    J : np.ndarray
        Weight matrix in V.
    K : np.ndarray
        Indegree matrix.
    N : np.ndarray
        Number of neurons in each population.
    tau_m : [float | np.narray]
        Membrane time constant in s.

    Returns
    -------
    np.ndarray
        Power spectrum in Hz. Shape: (len(freqs), len(populations)).
    """
    power = np.zeros(effective_connectivity.shape[0:2])
    for i, W in enumerate(effective_connectivity):
        Q = np.linalg.inv(np.identity(len(N)) - W)
        A = np.diag(np.ones(len(N))) * nu / N
        C = np.dot(Q, np.dot(A, np.transpose(np.conjugate(Q))))
        power[i] = np.absolute(np.diag(C))
    return power


def external_rates_for_fixed_input(network, mu_set, sigma_set, method='shift'):
    """
    Calculate external rates needed to get fixed mean and std input.

    Uses least square method to find best fitting solution for external rates
    such that the mean and standard deviation of the input to the neuronal
    populations is as close as possible to ``mu_set`` and ``sigma_set``.

    Generalization of equation E1 of :cite:t:`helias2013` and the corrected
    version in appendix F of :cite:t:`senk2020`.

    Parameters
    ----------
    network : nnmt.models.Network or child class instance.
        Network with the network parameters and previously calculated results
        listed in the following.
    mu_set : [float | np.array]
        Mean neuron activity in V.
    sigma_set : [float | np.array]
        Standard deviation of neuron activity in V.
    method : str
        Method used to calculate the target rates. Options: 'shift', 'taylor'.
        Default is 'shift'.

    Network parameters
    ------------------
    J : np.array
        Weight matrix in V.
    K : np.array
        Indegree matrix.
    V_th_rel : [float | np.array]
        Relative threshold potential in V.
    V_0_rel : [float | np.array]
        Relative reset potential in V.
    tau_m : [float | 1d array]
        Membrane time constant in s.
    tau_r : [float | 1d array]
        Refractory time in s.
    tau_s : [float | 1d array]
        Synaptic time constant in s.
    J_ext : np.array
        External weight matrix in V.
    K_ext : np.array
        Numbers of external input neurons to each population.

    Returns
    -------
    np.ndarray
        External rates in Hz.
    """

    list_of_params = ['J', 'K', 'V_0_rel', 'V_th_rel',
                      'tau_m', 'tau_r', 'tau_s',
                      'J_ext', 'K_ext']

    try:
        params = {key: network.network_params[key] for key in list_of_params}
    except KeyError as param:
        raise RuntimeError(
            f"You are missing {param} for calculating the effective "
            "connectivity!\n"
            "Have a look into the documentation for more details on 'lif' "
            "parameters.")

    params['mu_set'] = mu_set
    params['sigma_set'] = sigma_set
    params['method'] = method

    return _cache(network, _external_rates_for_fixed_input, params,
                  _prefix + 'external_rates_for_fixed_input')


@_check_positive_params
def _external_rates_for_fixed_input(mu_set, sigma_set,
                                    J, K, V_0_rel, V_th_rel,
                                    tau_m, tau_r, tau_s,
                                    J_ext, K_ext,
                                    method='shift'):
    """
    Calculate additional external excitatory and inhibitory Poisson input
    rates such that the input fixed by the mean and standard deviation
    is attained.
    Correction of equation E1 of :cite:t:`helias2013`. See appendix F of
    :cite:t:`senk2020`.

    Parameters
    ----------
    mu_set : [float | np.array]
        Mean neuron activity in V.
    sigma_set : [float | np.array]
        Standard deviation of neuron activity in V.
    J : np.array
        Weight matrix in V.
    K : np.array
        Indegree matrix.
    V_th_rel : [float | np.array]
        Relative threshold potential in V.
    V_0_rel : [float | np.array]
        Relative reset potential in V.
    tau_m : [float | 1d array]
        Membrane time constant in s.
    tau_r : [float | 1d array]
        Refractory time in s.
    tau_s : [float | 1d array]
        Synaptic time constant in s.
    J_ext : np.array
        External weight matrix in V.
    K_ext : np.array
        Numbers of external input neurons to each population.
    method : str
        Method used to calculate the target rates. Options: 'shift', 'taylor'.
        Default is 'shift'.

    Returns
    -------
    np.ndarray
        External rates in Hz.
    """
    # target rates for set mean and standard deviation of input
    if method == 'shift':
        target_rates = _firing_rate_shift(V_0_rel, V_th_rel,
                                          mu_set, sigma_set,
                                          tau_m, tau_r, tau_s)
    elif method == 'taylor':
        target_rates = _firing_rate_taylor(V_0_rel, V_th_rel,
                                           mu_set, sigma_set,
                                           tau_m, tau_r, tau_s)
    else:
        raise ValueError('Chosen method not implemented')

    # local only contributions
    mu_loc = _mean_input(target_rates, J, K, tau_m,
                         J_ext=0, K_ext=0, nu_ext=0)
    sigma_loc = _std_input(target_rates, J, K, tau_m,
                           J_ext=0, K_ext=0, nu_ext=0)

    # external working point that is to be achieved
    mu_ext = mu_set - mu_loc
    var_ext = sigma_set**2 - sigma_loc**2

    # the linear set of equations that needs to be solved
    LHS = np.append(K_ext * J_ext, K_ext * J_ext**2, axis=0)
    RHS = np.append(mu_ext / tau_m, var_ext / tau_m)

    # find a solution as good as possible using least square method
    nu_ext = np.linalg.lstsq(LHS, RHS, rcond=None)[0]

    if np.any(nu_ext < 0):
        raise RuntimeError(f'Negative rate detected: {nu_ext}')

    return nu_ext
