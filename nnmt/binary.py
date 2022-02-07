"""
Collection of functions for binarys neurons.

Network Functions
*****************

.. autosummary::
    :toctree: _toctree/lif/

    firing_rates
    balanced_threshold

Parameter Functions
*******************

.. autosummary::
    :toctree: _toctree/lif/

    _firing_rates
    _firing_rates_for_given_input
    _mean_input
    _std_input
    _balanced_threshold

"""
import numpy as np
from scipy.special import erfc as _erfc

from . import _solvers
from .utils import _cache


_prefix = 'binary.'


def _mean_input(m, J, K, J_ext=0, K_ext=0, m_ext=0):
    """
    Calculates the mean inputs in a network of binary neurons.

    Parameters
    ----------
    m : array
        Mean activity of each population.
    J : array
        Weight matrix.
    K : array
        Connectivity matrix.
    J_ext : array
        Weight matrix of external inputs.
    K_ext : array
        Connectivity matrix of external inputs.
    m_ext : float
        External input.

    Returns
    -------
    array
        Mean input of each population.
    """
    return np.dot(K * J, m) + np.dot(K_ext * J_ext, m_ext)


def _std_input(m, J, K, J_ext=0, K_ext=0, m_ext=0):
    """
    Calcs the standard deviation of the inputs in a network of binary neurons.

    Parameters
    ----------
    m : array
        Mean activity of each population.
    J : array
        Weight matrix.
    K : array
        Connectivity matrix.
    J_ext : array
        Weight matrix of external inputs.
    K_ext : array
        Connectivity matrix of external inputs.
    m_ext : float
        External input.

    Returns
    -------
    array
        Standard deviations of input.
    """
    return np.sqrt(np.dot(K * J**2, m * (1 - m))
                   + np.dot(K_ext * J_ext**2, m_ext * (1 - m_ext)))


def _firing_rates_for_given_input(mu, sigma, theta):
    """
    Calcs the firing rates of binary neurons for given input statistics.

    Parameters
    ----------
    mu : array
        Mean inputs for each population.
    sigma : array
        Standard deviation of inputs of each population.
    theta : [array | float]
        Firing thresholds for each population.

    Returns
    -------
    array
        Mean firing rates for each population.
    """
    return 0.5 * _erfc(-(mu - theta) / (np.sqrt(2) * sigma))


def firing_rates(network, **kwargs):
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
    array
        Mean firing rates for each population.
    """
    required_params = ['J', 'K', 'theta']
    optional_params = ['J_ext', 'K_ext', 'm_ext']

    try:
        params = {key: network.network_params[key] for key in required_params}
    except KeyError as param:
        raise RuntimeError(
            f"You are missing {param} for calculating the firing rate!\n"
            "Have a look into the documentation for more details on 'binary' "
            "parameters.")

    try:
        params = {key: network.network_params[key] for key in optional_params}
    except KeyError as param:
        pass

    params.update(kwargs)

    return _cache(network, _firing_rate, params, _prefix + 'firing_rates')


def _firing_rates(J, K, theta, **kwargs):
    """
    Calcs firing rates for each population in a network of binary neurons.

    See :func:`nnmt._solvers._firing_rate_integration` for integration
    procedure.

    Uses :func:`nnmt.binary._firing_rates_for_given_input`.

    Parameters
    ----------
    J : array
        Weight matrix.
    K : array
        Connectivity matrix.
    theta : [array | float]
        Firing threshold.

    Returns
    -------
    array
        Mean firing rates for each population.
    """
    firing_rate_params = {
        'theta': theta
    }
    input_funcs = [_mean_input, _std_input]
    input_params = {
        'J': J,
        'K': K,
    }

    return _solvers._firing_rate_integration(_firing_rate_for_given_input,
                                             firing_rate_params,
                                             input_funcs,
                                             input_params,
                                             **kwargs)


def balanced_threshold(network, m_exp):
    """
    Calculate threshold equal to input given expected mean activity (balance).

    See :func:`nnmt.binary._balanced_threshold` for full documentation.

    Parameters
    ----------
    network : nnmt.models.Network or child class instance.
        Network with the network parameters listed in the docstring of
        :func:`nnmt.binary._balanced_threshold`.
    m_exp : array
        Expected mean activity for each population.

    Returns
    -------
    array
        Balanced threshold for each population.
    """
    list_of_params = ['J', 'K']

    try:
        params = {key: network.network_params[key] for key in list_of_params}
    except KeyError as param:
        raise RuntimeError(
            f"You are missing {param} for calculating the balanced "
            "threshold!\nHave a look into the documentation for more details "
            "on 'binary' parameters.")

    params['m_exp'] = m_exp

    return _cache(network, _balanced_threshold, params,
                  _prefix + 'balanced_threshold')


def _balanced_threshold(m_exp, J, K):
    """
    Calculate threshold equal to input given expected mean activity (balance).

    Parameters
    ----------
    m_exp : array
        Expected mean activity for each population.
    J : array
        Weight matrix.
    K : array
        Connectivity matrix.

    Returns
    -------
    array
        Balanced threshold for each population.
    """
    return np.dot(K * J, m_exp)
