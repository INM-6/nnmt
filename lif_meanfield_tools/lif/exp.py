import warnings
import numpy as np
from scipy.special import (
    erf as _erf,
    zetac as _zetac
    )

from .. import ureg as ureg
from ..utils import (_check_positive_params,
                     _check_k_in_fast_synaptic_regime,
                     _check_and_store)

from . import _static
                      
from .delta import _firing_rate as _delta_firing_rate


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

    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    x_th = (np.sqrt(2.) * (V_th_rel - mu) / sigma)
    x_r = (np.sqrt(2.) * (V_0_rel - mu) / sigma)
    # this brings tau_m and tau_r into the correct vectorized form if they are
    # scalars and doesn't do anything if they are arrays of appropriate size
    V_0_rel = V_0_rel + x_th - x_th
    V_th_rel = V_th_rel + x_th - x_th
    tau_m = tau_m + x_th - x_th
    tau_s = tau_s + x_th - x_th
    tau_r = tau_r + x_th - x_th

    # preventing overflow in np.exponent in Phi(s)
    # note: this simply returns the white noise result... is this ok?
    result = np.zeros(len(mu))
    
    # do slightly different calculations for large thresholds
    overflow_mask = x_th > 20.0 / np.sqrt(2.)
    regular_mask = np.invert(overflow_mask)
    
    # white noise firing rate
    if np.any(overflow_mask):
        result[overflow_mask] = _delta_firing_rate(tau_m[overflow_mask],
                                                  tau_r[overflow_mask],
                                                  V_th_rel[overflow_mask],
                                                  V_0_rel[overflow_mask],
                                                  mu[overflow_mask],
                                                  sigma[overflow_mask])
    if np.any(regular_mask):
        result[regular_mask] = _delta_firing_rate(tau_m[regular_mask],
                                                 tau_r[regular_mask],
                                                 V_th_rel[regular_mask],
                                                 V_0_rel[regular_mask],
                                                 mu[regular_mask],
                                                 sigma[regular_mask])

    dPhi = _Phi(x_th[regular_mask]) - _Phi(x_r[regular_mask])
    
    # colored noise firing rate (might this lead to negative rates?)
    result[regular_mask] = (result[regular_mask]
                            - np.sqrt(tau_s[regular_mask]
                                      / tau_m[regular_mask])
                            * alpha
                            / (tau_m[regular_mask] * np.sqrt(2)) * dPhi
                            * (result[regular_mask] * tau_m[regular_mask])**2)
                            
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
