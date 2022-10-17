"""
This module contains utility functions, mainly used for internal purposes.

Cache
*****

.. autosummary::
    :toctree: _toctree/utils/

    _cache

Checks
******

.. autosummary::
    :toctree: _toctree/utils/

    check_if_positive
    check_for_valid_k_in_fast_synaptic_regime
    _check_positive_params
    _check_k_in_fast_synaptic_regime


Unit and Pint Related Functions
*******************************

.. autosummary::
    :toctree: _toctree/utils/

    pint_append
    pint_array
    pint_array_of_dimension_plus_one
    _strip_units
    _to_si_units
    _convert_to_si_and_strip_units
    _convert_from_si_to_prefixed
    _convert_from_prefixed_to_si

Miscellaneous
*************

.. autosummary::
    :toctree: _toctree/utils/

    build_full_arg_list
    get_list_of_required_parameters
    get_list_of_optional_parameters
    get_required_network_params
    get_optional_network_params
    get_required_results

"""

import inspect
import warnings
from functools import wraps
import numpy as np
import hashlib

from . import ureg


def _cache(network, func, params, result_keys, units=None):
    """
    Save result of ``func(**params)`` into network dicts using `result_keys`.

    This function serves as a wrapper for functions that calculate quantities
    which are to be stored in the network's result dicts. First it creates a
    hash using the function name, the passed parameters, and the result keys,
    and checks whether this hash is a key of the network's
    ``results_hash_dict``. If this is the case, the old result is returned.

    If not, the new result is calculated and stored in the
    ``results_hash_dict`` and the ``results`` dict. The unit of the returned
    result is stored in the network's ``result_units dict``. Then the new
    result is returned.

    Parameters
    ----------
    network : Network object or child class instance.
        The network whose dicts are used for storing the results.
    func : function
        Function whose return value should be cached.
    params : dict
        Parameters passed on to `func`.
    result_keys : str or list of str
        Specifies under which keys the result should be stored.
    units : str or list of str
        Units of results. Default is None.

    Returns
    -------
    ``func(**params)``
    """
    # make sure result keys are array
    # here we convert them to a list, because otherwise you might run into a
    # bug of the h5py_wrapper, which saves the type of the keys and after
    # converting them to a numpy array they are numpy strings
    # this then leads to a problem when loading the h5 file, because the
    # h5py_wrapper doesn't know the numpy string type.
    result_keys = np.atleast_1d(result_keys).tolist()

    # create unique hash for given function parameter combination
    label = str((func.__name__, result_keys, tuple(sorted(params.items()))))
    h = hashlib.md5(label.encode('utf-8')).hexdigest()

    # collect results
    results = getattr(network, 'results')
    results_hash_dict = getattr(network, 'results_hash_dict')

    # if corresponding result exists return cached value
    if h in results_hash_dict.keys():
        if len(result_keys) == 1:
            new_results = results_hash_dict[h][result_keys[0]]
        else:
            new_results = [results_hash_dict[h][key] for key in result_keys]
    # if corresponding result does not exists return newly calculated value
    else:
        # calculate new results
        new_results = func(**params)

        # create hash dict entry
        if len(result_keys) == 1:
            hash_dict = {result_keys[0]: new_results}
        else:
            assert len(result_keys) == len(new_results)
            hash_dict = dict(zip(result_keys, new_results))

        hash_dict['params'] = params
        # add entry to hash dict
        results_hash_dict[h] = hash_dict

    # update results
    if len(result_keys) == 1:
        results[result_keys[0]] = new_results
    else:
        for i, key in enumerate(result_keys):
            results[key] = new_results[i]

    # update network.results and network.results_hash_dict
    setattr(network, 'results', results)
    setattr(network, 'results_hash_dict', results_hash_dict)

    # update units
    if units is not None:
        units = np.atleast_1d(units).tolist()
        result_units = getattr(network, 'result_units')
        assert len(result_keys) == len(units)
        result_units.update(dict(zip(result_keys, units)))
        setattr(network, 'result_units', result_units)

    return new_results


def check_if_positive(parameters, parameter_names):
    """Check that will raise an error if parameters are negative."""
    for parameter, parameter_name in zip(parameters, parameter_names):
        try:
            if np.any(np.atleast_1d(parameter) < 0):
                raise ValueError('{} should be larger than zero!'.format(
                    parameter_name))
        except TypeError:
            if parameter < 0:
                raise ValueError('{} should be larger than zero!'.format(
                    parameter_name))


def _check_positive_params(func):
    """Decorator that checks that a fixed list of parameters is positive."""
    all_pos_params = ['K',
                      'K_ext',
                      'N',
                      'd_e',
                      'd_i',
                      'd_e_sd',
                      'd_i_sd',
                      'dimension',
                      'g',
                      'nu',
                      'nu_ext',
                      'nu_e_ext',
                      'nu_i_ext',
                      'sigma',
                      'tau_m',
                      'tau_s',
                      'tau_r',
                      'nu_ext',
                      ]

    @wraps(func)
    def decorator_check(*args, **kwargs):
        signature = inspect.signature(func)
        pos_param_names = [param for param in signature.parameters
                           if param in all_pos_params]
        all_args = build_full_arg_list(signature, args, kwargs)
        pos_params = [all_args[i] for i, param
                      in enumerate(signature.parameters)
                      if param in pos_param_names]
        check_if_positive(pos_params, pos_param_names)
        return func(*args, **kwargs)
    return decorator_check


def check_for_valid_k_in_fast_synaptic_regime(tau_m, tau_s):
    """ Check whether `tau_m` and `tau_s` imply fast synaptic regime."""
    k = np.atleast_1d(np.sqrt(tau_s / tau_m))
    if np.any((np.sqrt(0.1) < k)):
        k_warning = ('k=sqrt(tau_s/tau_m)={} might be too large for '
                     'calculation of firing rates via Taylor expansion!'
                     ).format(k)
        warnings.warn(k_warning)


def _check_k_in_fast_synaptic_regime(func):
    """
    Decorator checking whether `func` is operating in fast synaptic regime.
    """
    @wraps(func)
    def decorator_check(*args, **kwargs):
        signature = inspect.signature(func)
        args = build_full_arg_list(signature, args, kwargs)
        kwargs = dict(zip(signature.parameters, args))
        tau_m = kwargs['tau_m']
        tau_s = kwargs['tau_s']
        check_for_valid_k_in_fast_synaptic_regime(tau_m, tau_s)
        return func(**kwargs)
    return decorator_check


def pint_append(array, quantity, axis=0):
    """
    Append quantity to np.array quantity. Handles units correctly.

    Parameters
    ----------
    array : pint Quantity with np.array magnitude or just np.array
        Array to which quantity should be appended.
    quantity : pint Quantity or just something unitless
        Quantity which should be appended to array.
    axis : num
        Axis along which to append quantity to array.

    Returns
    -------
    pint Quantity with np.array magnitude
    """
    if isinstance(quantity, ureg.Quantity):
        return np.append(array.magnitude,
                         [quantity.magnitude],
                         axis=axis) * array.units
    else:
        return np.append(array, [quantity], axis=axis)


def pint_array(quantity_list):
    """
    Create quantity with magnitude np.array. Handles units correctly.

    Parameters
    ----------
    quantity_list : list
        List of quantities.
    """
    try:
        mags = [q.magnitude for q in quantity_list]
        unit = quantity_list[0].units
        array = np.array(mags) * unit
    except AttributeError:
        array = np.array(quantity_list)
    return array


def pint_array_of_dimension_plus_one(quantity):
    """
    Create quantity with magnitude np.array with one more dimension

    Handles units correctly.
    """
    if isinstance(quantity, ureg.Quantity):
        return np.array([quantity.magnitude]) * quantity.units
    else:
        return np.array([quantity])


def _strip_units(dict):
    """
    Strip units of quantities.
    """
    for key, item in dict.items():
        try:
            dict[key] = item.magnitude
        except AttributeError:
            pass


def _to_si_units(dict):
    """
    Convert dict of quantities to si units.
    """
    for key, item in dict.items():
        try:
            dict[key] = item.to_base_units()
        except AttributeError:
            pass


def _convert_to_si_and_strip_units(dict):
    """Converts dict of quantities to si units and strips them."""
    _to_si_units(dict)
    _strip_units(dict)


def _convert_from_si_to_prefixed(magnitude, unit):
    """ Converts a SI magnitude to the given unit. """
    try:
        base_unit = ureg.parse_unit_name(unit)[0][1]
    except IndexError:
        base_unit = str(ureg(unit).to_base_units().units)
    quantity = ureg.Quantity(magnitude, base_unit)
    quantity.ito(unit)
    return quantity


def _convert_from_prefixed_to_si(magnitude, unit):
    """
    Converts a given unit magnitude to the corresponding SI unit magnitude.
    """
    try:
        base_unit = ureg.parse_unit_name(unit)[0][1]
    except IndexError:
        base_unit = str(ureg(unit).to_base_units().units)
    quantity = ureg.Quantity(magnitude, unit)
    quantity.ito(base_unit)
    return quantity


def build_full_arg_list(signature, args, kwargs):
    """
    Creates a full list of arguments including standard arguments.

    Parameters
    ----------
    signature : Signature object
        The signature of a given function.
    args : list
        List of passed positional arguments.
    kwargs : dict
        Dict of passed keyword arguments.

    Returns
    -------
    list
        Full list of arguments.
    """

    keys = list(signature.parameters.keys())[len(args):]
    defaults = [param.default for param
                in signature.parameters.values()][len(args):]

    full_list = list(args)
    for key, default in zip(keys, defaults):
        if key in kwargs.keys():
            full_list.append(kwargs[key])
        else:
            full_list.append(default)

    return full_list


def get_list_of_required_parameters(func):
    """Returns list of arguments required by `func`."""
    sig = inspect.signature(func)
    required_params = [name for name, param in sig.parameters.items()
                       if param.default is param.empty]
    if 'args' in required_params:
        required_params.remove('args')
    if 'kwargs' in required_params:
        required_params.remove('kwargs')
    return required_params


def get_list_of_optional_parameters(func):
    """Returns list of optional arguments of `func`."""
    sig = inspect.signature(func)
    return [name for name, param in sig.parameters.items()
            if not param.default is param.empty]


def get_required_network_params(network, func, exclude=None):
    """
    Extracts dict with required args for `func` from `network.network_params`.
    """
    list_of_params = get_list_of_required_parameters(func)
    if exclude:
        for key in exclude:
            list_of_params.remove(key)
    try:
        params = {key: network.network_params[key] for key in list_of_params}
    except KeyError as param:
        raise RuntimeError(f'You are missing {param} for this calculation.')
    return params


def get_optional_network_params(network, func):
    """
    Extracts dict with optional args for `func` from `network.network_params`.

    Returns empty dict if any of the optional params is not found.
    """
    list_of_params = get_list_of_optional_parameters(func)
    list_of_params = (
        set(list_of_params).intersection(set(network.network_params.keys())))
    optional_params = {key: network.network_params[key]
                       for key in list_of_params}
    return optional_params


def get_required_results(network, keys, results_keys):
    """
    Extracts dict with results from `network.results`.

    Parameters
    ----------
    network : Network object or child class instance.
        The network whose dicts are used for storing the results.
    keys : list
        The keys used in the returned dictionary.
    results_keys : list
        The corresponding keys used in `network.results`.

    Returns
    -------
    dict
        The dictionary with the requested results using the given `keys`.
    """
    try:
        results = {key: network.results[rkey]
                   for key, rkey in zip(keys, results_keys)}
    except KeyError as quantity:
        raise RuntimeError(f'You first need to calculate the {quantity}.')
    return results


