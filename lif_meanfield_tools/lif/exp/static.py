from scipy.special import erf, zetac, lambertw, erfcx, dawsn, roots_legendre
import numpy as np
import mpmath

from ... import ureg
from ...utils import (check_if_positive,
                      check_for_valid_k_in_fast_synaptic_regime)

from ..static import _firing_rate_integration
from ..delta.static import _firing_rate as delta_firing_rate


def firing_rates(network, method='shift', integration_method='scef'):
    list_of_firing_rate_params = ['tau_m', 'tau_s', 'tau_r', 'V_th_rel',
                                  'V_0_rel']
    list_of_input_params = ['K', 'J', 'j', 'tau_m', 'nu_ext', 'K_ext', 'g',
                            'nu_e_ext', 'nu_i_ext']
    try:
        firing_rate_params = {key: network.network_params[key]
                              for key in list_of_firing_rate_params}
        input_params = {key: network.network_params[key]
                        for key in list_of_input_params}
    except KeyError as param:
        print(f"You are missing {param} for calculating the firing rate!\n"
              "Have a look into the documentation for more details on 'lif'"
              " parameters.")
        
    firing_rate_params['method'] = method
    firing_rate_params['integration_method'] = integration_method
        
    return _firing_rate_integration(_firing_rate,
                                    firing_rate_params,
                                    input_params)


def _firing_rate(tau_m, tau_s, tau_r, V_th_rel, V_0_rel, mu, sigma,
                 method='shift', integration_method='scef'):
    """
    Calcs stationary firing rates for exp PSCs

    Calculates the stationary firing rate of a neuron with synaptic filter of
    time constant tau_s driven by Gaussian noise with mean mu and standard
    deviation sigma, using Eq. 4.33 in Fourcaud & Brunel (2002) with Taylor
    expansion around k = sqrt(tau_s/tau_m).

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
    
    Returns:
    --------
    float:
        Stationary firing rate in Hz.
    """
    pos_parameters = [tau_m, tau_s, tau_r, sigma]
    pos_parameter_names = ['tau_m', 'tau_s', 'tau_r', 'sigma']
    check_if_positive(pos_parameters, pos_parameter_names)
    if np.any(V_th_rel - V_0_rel < 0):
        raise ValueError('V_th should be larger than V_0!')
    
    check_for_valid_k_in_fast_synaptic_regime(tau_m, tau_s)
    
    if method == 'taylor':
        return _firing_rate_taylor(
            tau_m, tau_s, tau_r, V_th_rel, V_0_rel, mu, sigma,
            integration_method)
    elif method == 'shift':
        return _firing_rate_shift(
            tau_m, tau_s, tau_r, V_th_rel, V_0_rel, mu, sigma,
            integration_method)
    else:
        raise ValueError('Method not implemented.')
    

def _firing_rate_taylor(tau_m, tau_s, tau_r, V_th_rel, V_0_rel, mu, sigma,
                        integration_method):
    """Helper function implementing nu0_fb433 without quantities."""
    # use zetac function (zeta-1) because zeta is not giving finite values for
    # arguments smaller 1.
    alpha = np.sqrt(2.) * abs(zetac(0.5) + 1)

    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)

    # preventing overflow in np.exponent in Phi(s)
    # note: this simply returns the white noise result... is this ok?
    result = np.zeros(len(mu)) * ureg.Hz
    for i, (mu, sigma) in enumerate(zip(mu, sigma)):
        # additional prefactor sqrt(2) because its x from Schuecker 2015
        x_th = (np.sqrt(2.) * (V_th_rel - mu) / sigma).magnitude
        x_r = (np.sqrt(2.) * (V_0_rel - mu) / sigma).magnitude
        if x_th > 20.0 / np.sqrt(2.):
            result[i] = delta_firing_rate(tau_m, tau_r, V_th_rel, V_0_rel, mu,
                                          sigma, integration_method)
        else:
            # white noise firing rate
            r = delta_firing_rate(tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma,
                                  integration_method)

            dPhi = _Phi(x_th) - _Phi(x_r)
            # colored noise firing rate (might this lead to negative rates?)
            result[i] = (r - np.sqrt(tau_s / tau_m) * alpha
                         / (tau_m * np.sqrt(2))
                         * dPhi * (r * tau_m)**2)

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
                                  * (1 + erf(s / np.sqrt(2))))


def _firing_rate_shift(tau_m, tau_s, tau_r, V_th_rel, V_0_rel, mu, sigma,
                       integration_method):
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
    V_th_rel: float
        Relative threshold potential in mV.
    V_0_rel: float
        Relative reset potential in mV.
    mu: float
        Mean neuron activity in mV.
    sigma:
        Standard deviation of neuron activity in mV.

    Returns:
    --------
    float:
        Stationary firing rate in Hz.
    """
    # using zetac (zeta-1), because zeta is giving nan result for arguments
    # smaller 1
    alpha = np.sqrt(2) * abs(zetac(0.5) + 1)
    # effective threshold
    # additional factor sigma is canceled in siegert
    V_th1 = V_th_rel + sigma * alpha / 2. * np.sqrt(tau_s / tau_m)
    # effective reset
    V_01 = V_0_rel + sigma * alpha / 2. * np.sqrt(tau_s / tau_m)
    # use standard Siegert with modified threshold and reset
    return delta_firing_rate(tau_m, tau_r, V_th1, V_01, mu, sigma,
                             integration_method)
