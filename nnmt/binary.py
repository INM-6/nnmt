# ToDo
# - transfer code to NNMT:
# - generalize network model
# - abstract solver

import numpy as np
from scipy.special import erfc as _erfc

from . import _solvers
from .utils import _cache


_prefix = 'binary.'


def _mean_input(m, NE, NI, K, w, g):
    """
    Calculates the mean input in a network of binary neurons.

    Parameters
    ----------
    m : float
        Mean activity / magnetization.
    NE : int
        Number of excitatory neurons.
    NI : int
        Number of inhibitory neurons.
    K : int
        Number of inputs.
    w : float
        Synaptic weight.
    g : float
        Ratio of inhibitory to excitatory weights.

    Returns
    -------
    float
        Mean input.
    """
    return K * w * (1 - g * NI/NE) * m


def _std_input(m, NE, NI, K, w, g):
    """
    Calcs the standard deviation of the input in a network of binary neurons.

    Parameters
    ----------
    m : float
        Mean activity / magnetization.
    NE : int
        Number of excitatory neurons.
    NI : int
        Number of inhibitory neurons.
    K : int
        Number of inputs.
    w : float
        Synaptic weight.
    g : float
        Ratio of inhibitory to excitatory weights.

    Returns
    -------
    float
        Mean input.
    """
    return np.sqrt(K * w**2 * (1 + g**2 * NI/NE) * m * (1 - m))


def _firing_rate_for_given_input(mu, sigma, theta):
    """
    Calcs the firing rate of binary neurons for given input statistics.

    Parameters
    ----------
    mu : float
        Mean input.
    sigma : float
        Standard deviation of input.
    theta : float
        Firing threshold.

    Returns
    -------
    float
        Mean firing rate.
    """
    return 0.5 * _erfc(-(mu - theta) / (np.sqrt(2) * sigma))


def firing_rate(network, **kwargs):
    """
    Calculates stationary firing rates for a network of binary neurons.

    See :func:`nnmt.binary._firing_rates` for full documentation.

    Parameters
    ----------
    network : nnmt.models.Network or child class instance.
        Network with the network parameters listed in the docstring of
        :func:`nnmt.binary._firing_rates`.
    kwargs
        For additional kwargs regarding the fixpoint iteration procedure see
        :func:`nnmt._solvers._firing_rate_integration`.

    Returns
    -------
    float
        Mean firing rate of binary neurons.
    """
    list_of_params = ['NE', 'NI', 'K', 'w', 'g', 'theta']

    try:
        params = {key: network.network_params[key] for key in list_of_params}
    except KeyError as param:
        raise RuntimeError(
            f"You are missing {param} for calculating the firing rate!\n"
            "Have a look into the documentation for more details on 'binary' "
            "parameters.")

    params.update(kwargs)

    return _cache(network, _firing_rate, params, _prefix + 'firing_rates')


def _firing_rate(NE, NI, K, w, g, theta, **kwargs):
    """
    Calculation of firing rates for a network of binary neurons.

    See :func:`nnmt._solvers._firing_rate_integration` for integration
    procedure.

    Uses :func:`nnmt.binary._firing_rate_for_given_input`.
    """
    firing_rate_params = {
        'theta': theta
    }
    input_funcs = [_mean_input, _std_input]
    input_params = {
        'NE': NE,
        'NI': NI,
        'K': K,
        'w': w,
        'g': g,
    }

    return _solvers._firing_rate_integration(_firing_rate_for_given_input,
                                             firing_rate_params,
                                             input_funcs,
                                             input_params,
                                             **kwargs)