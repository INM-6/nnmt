"""
Collection of functions for binarys neurons.

Network Functions
*****************

.. autosummary::
    :toctree: _toctree/lif/

    mean_activity
    mean_input
    std_input
    working_point
    balanced_threshold

Parameter Functions
*******************

.. autosummary::
    :toctree: _toctree/lif/

    _mean_activity
    _mean_activity_for_given_input
    _mean_input
    _std_input
    _balanced_threshold

"""
import numpy as np
from scipy.special import erfc as _erfc

from . import _solvers
from .utils import _cache


_prefix = 'binary.'


def _mean_activity_for_given_input(mu, sigma, theta):
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


def mean_activity(network, **kwargs):
    """
    Calculates stationary firing rates for a network of binary neurons.

    See :func:`nnmt.binary._mean_activity` for full documentation.

    Parameters
    ----------
    network : nnmt.models.Network or child class instance.
        Network with the network parameters listed in the docstring of
        :func:`nnmt.binary._mean_activity`.
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

    return _cache(network, _mean_activity, params, _prefix + 'mean_activity')


def _mean_activity(J, K, theta, **kwargs):
    """
    Calcs firing rates for each population in a network of binary neurons.

    See :func:`nnmt._solvers._firing_rate_integration` for integration
    procedure.

    Uses :func:`nnmt.binary._mean_activity_for_given_input`.

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
    input_params = {
        'J': J,
        'K': K,
    }
    input_dict = dict(
        mu={'func': _mean_input,
            'params': input_params},
        sigma={'func': _std_input,
               'params': input_params},
    )

    return _solvers._firing_rate_integration(_mean_activity_for_given_input,
                                             firing_rate_params,
                                             input_dict,
                                             **kwargs)


def mean_input(network):
    '''
    Calc mean inputs to populations as function of firing rates of populations.

    See :func:`nnmt.binary._mean_input` for full documentation.

    Parameters
    ----------
    network : Network object
        Model with the network parameters and previously calculated results
        listed in :func:`nnmt.binary._mean_input`.

    Returns
    -------
    array
        Array of mean inputs to each population.
    '''
    required_params = ['J', 'K']
    optional_params = ['J_ext', 'K_ext', 'm_ext']

    try:
        params = {key: network.network_params[key] for key in required_params}
    except KeyError as param:
        raise RuntimeError(f'You are missing {param} for this calculation.')
    try:
        params = {key: network.network_params[key] for key in optional_params}
    except KeyError as param:
        pass

    try:
        params['m'] = network.results[_prefix + 'mean_activity']
    except KeyError as quantity:
        raise RuntimeError(f'You first need to calculate the {quantity}.')

    return _cache(network, _mean_input, params, _prefix + 'mean_input')


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


def std_input(network):
    '''
    Calcs the standard deviation of the inputs in a network of binary neurons.

    See :func:`nnmt.binary._std_input` for full documentation.

    Parameters
    ----------
    network : Network object
        Model with the network parameters and previously calculated results
        listed in :func:`nnmt.binary._std_input`.

    Returns
    -------
    array
        Array of standard deviations of inputs to each population.
    '''
    required_params = ['J', 'K']
    optional_params = ['J_ext', 'K_ext', 'm_ext']

    try:
        params = {key: network.network_params[key] for key in required_params}
    except KeyError as param:
        raise RuntimeError(f'You are missing {param} for this calculation.')
    try:
        params = {key: network.network_params[key] for key in optional_params}
    except KeyError as param:
        pass

    try:
        params['m'] = network.results[_prefix + 'mean_activity']
    except KeyError as quantity:
        raise RuntimeError(f'You first need to calculate the {quantity}.')

    return _cache(network, _std_input, params, _prefix + 'std_input')


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


def working_point(network, **kwargs):
    """
    Calculates working point (rates, mean, and std input) for binary network.

    Calculates the firing rates using :func:`nnmt.binary.mean_activity`,
    the mean input using :func:`nnmt.binary.mean_input`,
    and the standard deviation of the input using
    :func:`nnmt.binary.std_input`.

    Parameters
    ----------
    network : nnmt.models.Network or child class instance.
        Network with the network parameters listed in
        :func:`nnmt.binary._mean_activity`.
    kwargs
        For additional kwargs regarding the fixpoint iteration procedure see
        :func:`nnmt._solvers._firing_rate_integration`.

    Returns
    -------
    dict
        Dictionary containing firing rates, mean input and std input.
    """
    return {'mean_activity': mean_activity(network, **kwargs),
            'mean_input': mean_input(network),
            'std_input': std_input(network)}


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
