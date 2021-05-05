import warnings
import numpy as np
from scipy.special import (
    erf as _erf,
    zetac as _zetac,
    erfcx as _erfcx,
    )

from .. import ureg as ureg
from ..utils import (_check_positive_params,
                     _check_k_in_fast_synaptic_regime,
                     _check_and_store)

from . import _static
                      
from .delta import (
    _firing_rate as _delta_firing_rate,
    _get_erfcx_integral_gl_order,
    _siegert_exc,
    _siegert_inh,
    _siegert_interm,
    )


_prefix = 'lif.exp.'


@_check_and_store(_prefix, ['firing_rates'], ['firing_rates_method'])
def firing_rates(network, method='shift'):
    """
    Calculates stationary firing rates for exp PSCs.

    Calculates the stationary firing rate of a neuron with synaptic filter of
    time constant tau_s driven by Gaussian noise with mean mu and standard
    deviation sigma, using Eq. 4.33 in Fourcaud & Brunel (2002) with Taylor
    expansion around k = sqrt(tau_s/tau_m).

    Parameters:
    -----------
    network: lif_meanfield_tools.create.Network or child class instance.
        Network with the network parameters listed in the following.
    method: str
        Method used to integrate the adapted Siegert function. Options: 'shift'
        or 'taylor'. Default is 'shift'.
    
    Network parameters:
    -------------------
    tau_m: float
        Membrane time constant in s.
    tau_s: float
        Synaptic time constant in s.
    tau_r: float
        Refractory time in s.
    V_0_rel: float
        Relative reset potential in V.
    V_th_rel: float
        Relative threshold potential in V.
    K: np.array
        Indegree matrix.
    J: np.array
        Weight matrix in V.
    j: float
        Synaptic weight in V.
    nu_ext: float
        Firing rate of external input in Hz.
    K_ext: np.array
        Numbers of external input neurons to each population.
    g: float
        Relative inhibitory weight.
    nu_e_ext: float
        Firing rate of additional external excitatory Poisson input in Hz.
    nu_i_ext: float
        Firing rate of additional external inhibitory Poisson input in Hz.

    Returns:
    --------
    Quantity(np.array, 'hertz')
        Array of firing rates of each population in Hz.
    """
    list_of_firing_rate_params = ['tau_m', 'tau_s', 'tau_r', 'V_th_rel',
                                  'V_0_rel']
    list_of_input_params = ['K', 'J', 'tau_m', 'nu_ext', 'K_ext', 'J_ext',
                            'tau_m_ext']

    try:
        firing_rate_params = {key: network.network_params[key]
                              for key in list_of_firing_rate_params}
        input_params = {key: network.network_params[key]
                        for key in list_of_input_params}
    except KeyError as param:
        raise RuntimeError(
            f"You are missing {param} for calculating the firing rate!\n"
            "Have a look into the documentation for more details on 'lif' "
            "parameters.")
    
    if method == 'shift':
        return _static._firing_rate_integration(_firing_rate_shift,
                                                firing_rate_params,
                                                input_params) * ureg.Hz
    elif method == 'taylor':
        return _static._firing_rate_integration(_firing_rate_taylor,
                                                firing_rate_params,
                                                input_params) * ureg.Hz


@_check_positive_params
@_check_k_in_fast_synaptic_regime
def _firing_rate(tau_m, tau_s, tau_r, V_th_rel, V_0_rel, mu, sigma,
                 method='shift'):
    """
    Calcs stationary firing rates for exp PSCs

    See `firing_rates` for full documentation.

    Parameters:
    -----------
    tau_m: float
        Membrane time constant in seconds.
    tau_s: float
        Synaptic time constant in seconds.
    tau_r: float
        Refractory time in seconds.
    V_th_rel: float
        Relative threshold potential in mV.
    V_0_rel: float
        Relative reset potential in mV.
    mu: float
        Mean neuron activity in mV.
    sigma: float
        Standard deviation of neuron activity in mV.
    method: str
        Method used for adjusting the Siegert functions for exponentially
        shaped post synaptic currents. Options: 'shift', 'taylor'. Default is
        'shift'.
    
    Returns:
    --------
    float:
        Stationary firing rate in Hz.
    """
    if np.any(V_th_rel - V_0_rel < 0):
        raise ValueError('V_th should be larger than V_0!')
    
    if method == 'taylor':
        return _firing_rate_taylor(
            tau_m, tau_s, tau_r, V_th_rel, V_0_rel, mu, sigma)
    elif method == 'shift':
        return _firing_rate_shift(
            tau_m, tau_s, tau_r, V_th_rel, V_0_rel, mu, sigma)
    else:
        raise ValueError(f'Method {method} not implemented.')
    

@_check_positive_params
@_check_k_in_fast_synaptic_regime
def _firing_rate_taylor(tau_m, tau_s, tau_r, V_th_rel, V_0_rel, mu, sigma):
    """
    Calcs stationary firing rates for exp PSCs using a Taylor expansion.

    See `firing_rates` for full documentation.

    Parameters:
    -----------
    tau_m: float
        Membrane time constant in seconds.
    tau_s: float
        Synaptic time constant in seconds.
    tau_r: float
        Refractory time in seconds.
    V_th_rel: float or np.array
        Relative threshold potential in mV.
    V_0_rel: float or np.array
        Relative reset potential in mV.
    mu: float or np.array
        Mean neuron activity in mV.
    sigma: float or np.array
        Standard deviation of neuron activity in mV.
    
    Returns:
    --------
    float or np.array:
        Stationary firing rate in Hz.
    """
    # use _zetac function (zeta-1) because zeta is not giving finite values for
    # arguments smaller 1.
    alpha = np.sqrt(2.) * abs(_zetac(0.5) + 1)

    nu0 = _delta_firing_rate(tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma)
    nu0_dPhi = _nu0_dPhi(tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma)
    result = nu0 * (1 - np.sqrt(tau_s * tau_m / 2) * alpha * nu0_dPhi)
    if np.any(result < 0):
        warnings.warn("Negative firing rates detected. You might be in an "
                      "invalid regime. Use `method='shift'` for "
                      "calculating the firing rates instead.")
        
    if result.shape == (1,):
        return result.item(0)
    else:
        return result


def _Phi(s):
    """
    helper function to calculate stationary firing rates with synaptic
    filtering

    corresponds to u^-2 F in Eq. 53 of the following publication


    Schuecker, J., Diesmann, M. & Helias, M.
    Reduction of colored noise in excitable systems to white
    noise and dynamic boundary conditions. 1–23 (2014).
    """
    return np.sqrt(np.pi / 2.) * (np.exp(s**2 / 2.)
                                  * (1 + _erf(s / np.sqrt(2))))
    

def _Phi_neg(s):
    """Calculate Phi(s) for negative arguments"""
    assert np.all(s <= 0)
    return np.sqrt(np.pi / 2.) * _erfcx(np.abs(s) / np.sqrt(2))


def _Phi_pos(s):
    """Calculate Phi(s) without exp(-s**2 / 2) factor for positive arguments"""
    assert np.all(s >= 0)
    return np.sqrt(np.pi / 2.) * (2 - np.exp(-s**2 / 2.)
                                  * _erfcx(s / np.sqrt(2)))


def _nu0_dPhi(tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma):
    """Calculate nu0 * ( Phi(sqrt(2)*y_th) - Psi(sqrt(2)*y_r) ) safely."""
    y_th = (V_th_rel - mu) / sigma
    y_r = (V_0_rel - mu) / sigma
    # bring into appropriate shape
    y_th = np.atleast_1d(y_th)
    y_r = np.atleast_1d(y_r)
    # this brings tau_m and tau_r into the correct vectorized form if they are
    # scalars and doesn't do anything if they are arrays of appropriate size
    tau_m = tau_m + y_th - y_th
    tau_r = tau_r + y_th - y_th
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


@_check_positive_params
@_check_k_in_fast_synaptic_regime
def _firing_rate_shift(tau_m, tau_s, tau_r, V_th_rel, V_0_rel, mu, sigma):
    """
    Calculates stationary firing rates including synaptic filtering.

    Based on Fourcaud & Brunel 2002, using shift of the integration boundaries
    in the white noise Siegert formula, as derived in Schuecker 2015.

    Parameters:
    -----------
    tau_m: float
        Membrane time constant in seconds.
    tau_s: float
        Synaptic time constant in seconds.
    tau_r: float
        Refractory time in seconds.
    V_th_rel: float or np.array
        Relative threshold potential in mV.
    V_0_rel: float or np.array
        Relative reset potential in mV.
    mu: float or np.array
        Mean neuron activity in mV.
    sigma: float or np.array
        Standard deviation of neuron activity in mV.

    Returns:
    --------
    float or np.array:
        Stationary firing rate in Hz.
    """
    # using _zetac (zeta-1), because zeta is giving nan result for arguments
    # smaller 1
    alpha = np.sqrt(2) * abs(_zetac(0.5) + 1)
    # effective threshold
    # additional factor sigma is canceled in siegert
    V_th1 = V_th_rel + sigma * alpha / 2. * np.sqrt(tau_s / tau_m)
    # effective reset
    V_01 = V_0_rel + sigma * alpha / 2. * np.sqrt(tau_s / tau_m)
    # use standard Siegert with modified threshold and reset
    return _delta_firing_rate(tau_m, tau_r, V_th1, V_01, mu, sigma)


@_check_and_store(_prefix, ['mean_input'])
def mean_input(network):
    '''
    Calc mean inputs to populations as function of firing rates of populations.

    Following Fourcaud & Brunel (2002).
    
    Parameters:
    -----------
    network: lif_meanfield_tools.create.Network or child class instance.
        Network with the network parameters and previously calculated results
        listed in the following.
        
    Network results:
    ----------------
    nu: Quantity(np.array, 'hertz')
        Firing rates of populations in Hz.
    
    Network parameters:
    -------------------
    tau_m: float
        Membrane time constant in s.
    K: np.array
        Indegree matrix.
    J: np.array
        Weight matrix in V.
    j: float
        Synaptic weight in V.
    nu_ext: float
        Firing rate of external input in Hz.
    K_ext: np.array
        Numbers of external input neurons to each population.
    g: float
        Relative inhibitory weight.
    nu_e_ext: float
        Firing rate of additional external excitatory Poisson input in Hz.
    nu_i_ext: float
        Firing rate of additional external inhibitory Poisson input in Hz.

    Returns:
    --------
    Quantity(np.array, 'volt')
        Array of mean inputs to each population in V.
    '''
    return _static._mean_input(network, _prefix)


@_check_and_store(_prefix, ['std_input'])
def std_input(network):
    '''
    Calc standard deviation of inputs to populations.
    
    Following Fourcaud & Brunel (2002).
    
    Parameters:
    -----------
    network: lif_meanfield_tools.create.Network or child class instance.
        Network with the network parameters and previously calculated results
        listed in the following.
        
    Network results:
    ----------------
    nu: Quantity(np.array, 'hertz')
        Firing rates of populations in Hz.
    
    Network parameters:
    -------------------
    tau_m: float
        Membrane time constant in s.
    K: np.array
        Indegree matrix.
    J: np.array
        Weight matrix in V.
    j: float
        Synaptic weight in V.
    nu_ext: float
        Firing rate of external input in Hz.
    K_ext: np.array
        Numbers of external input neurons to each population.
    g: float
        Relative inhibitory weight.
    nu_e_ext: float
        Firing rate of additional external excitatory Poisson input in Hz.
    nu_i_ext: float
        Firing rate of additional external inhibitory Poisson input in Hz.

    Returns:
    --------
    Quantity(np.array, 'volt')
        Array of mean inputs to each population in V.
    '''
    return _static._std_input(network, _prefix)
