"""
This module contains helper functions.
"""

import inspect
import warnings
from functools import wraps
import numpy as np
from decorator import decorator
import hashlib

from . import ureg


def _check_and_store(prefix, result_keys, analysis_keys=None):
    """
    Decorator function that checks whether result are already existing.

    This decorator serves as a wrapper for functions that calculate
    quantities which are to be stored in self.results. First it checks,
    whether the result already has been stored in self.results. If this is
    the case, it returns that result. If not, the calculation is executed,
    the result is stored in self.results and the result is returned.
    Additionally results are stored in self.results_hash_dict to simplify
    searching.

    If the wrapped function gets additional parameters passed, one should
    also include an analysis key, under which the new analysis parameters
    should be stored in the dictionary self.analysis_params. Then, the
    decorator first checks, whether the given parameters have been used
    before and returns the corresponding results.
    
    This function can only handle unitless objects or quantities. Lists or
    arrays of quantites are not allowed. Use quantity arrays instead (a
    quantity with array magnitude and a unit).

    TODO: Implement possibility to pass list of result_keys

    Parameters:
    -----------
    result_keys: list
        Specifies under which keys the result should be stored.
    analysis_key: list
        Specifies under which keys the analysis_parameters should be
        stored.

    Returns:
    --------
    func
        decorator function
    """
    # add prefix
    result_keys = [prefix + key for key in result_keys]
    if analysis_keys is not None:
        analysis_keys = [prefix + key for key in analysis_keys]

    @decorator
    def decorator_check_and_store(func, network, *args, **kwargs):
        """ Decorator with given parameters, returns expected results. """
        # collect analysis_params
        analysis_params = getattr(network, 'analysis_params')

        # collect results
        results = getattr(network, 'results')
        results_hash_dict = getattr(network, 'results_hash_dict')

        new_params = []
        if analysis_keys is not None:
            # add not passed standard arguments to args:
            # need to add network first because function signature expects
            # network at first position
            args = list(args)
            args.insert(0, network)
            args = build_full_arg_list(inspect.signature(func),
                                       args, kwargs)
            # remove network
            args.pop(0)
            # empty kwargs which are now included in args
            kwargs = {}
            for i, key in enumerate(analysis_keys):
                new_params.append(args[i])

        # calculate hash from result and analysis keys and analysis params
        label = str(result_keys) + str(analysis_keys) + str(new_params)
        h = hashlib.md5(label.encode('utf-8')).hexdigest()
        # check if hash already exists and get corresponding result
        if h in results_hash_dict.keys():
            # if only one key is present don't use list
            if len(result_keys) == 1:
                new_results = results_hash_dict[h][result_keys[0]]
            else:
                new_results = [results_hash_dict[h][key]
                               for key in result_keys]
        else:
            # if not, calculate new result
            new_results = func(network, *args, **kwargs)

            # create new hash dict entry
            if len(result_keys) == 1:
                hash_dict = {result_keys[0]: new_results}
            else:
                hash_dict = {key: new_results[i] for i, key
                             in enumerate(result_keys)}
            if analysis_keys:
                analysis_dict = {}
                for key, param in zip(analysis_keys, new_params):
                    analysis_dict[key] = param
                hash_dict['analysis_params'] = analysis_dict
            results_hash_dict[h] = hash_dict
            
        # create new results and analysis_params entries
        if len(result_keys) == 1:
            results[result_keys[0]] = new_results
        else:
            for i, key in enumerate(result_keys):
                results[key] = new_results[i]
        if analysis_keys:
            analysis_dict = {}
            for key, param in zip(analysis_keys, new_params):
                analysis_params[key] = param
            
        # update network.results and network.results_hash_dict
        setattr(network, 'results', results)
        setattr(network, 'results_hash_dict', results_hash_dict)
        setattr(network, 'analysis_params', analysis_params)

        # return new result
        return new_results

    return decorator_check_and_store


def check_if_positive(parameters, parameter_names):
    """Check that will raise an error if parameters are negative."""
    for parameter, parameter_name in zip(parameters, parameter_names):
        try:
            if any(p < 0 for p in pint_array(parameter).flatten()):
                raise ValueError('{} should be larger than zero!'.format(
                    parameter_name))
        except TypeError:
            if parameter < 0:
                raise ValueError('{} should be larger than zero!'.format(
                    parameter_name))


def check_positive_params(func):
    all_pos_params = ['C',
                      'K',
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
        if len(args) != 0:
            pos_params = [args[i] for i, param
                          in enumerate(signature.parameters)
                          if param in pos_param_names]
        else:
            pos_params = [kwargs[param] for param in pos_param_names]
        check_if_positive(pos_params, pos_param_names)
        return func(*args, **kwargs)
    return decorator_check


def check_for_valid_k_in_fast_synaptic_regime(tau_m, tau_s):
    """ Check whether we are in fast synaptic regime."""
    k = np.sqrt(tau_s / tau_m)
    if (np.sqrt(0.1) < k) & (k < 1):
        k_warning = ('k=sqrt(tau_s/tau_m)={} might be too large for '
                     'calculation of firing rates via Taylor expansion!'
                     ).format(k)
        warnings.warn(k_warning)
    if 1 <= k:
        raise ValueError('k=sqrt(tau_s/tau_m) is too large for calculation of '
                         'firing rates via Taylor expansion!')
        
        
def check_k_in_fast_synaptic_regime(func):
    @wraps(func)
    def decorator_check(*args, **kwargs):
        signature = inspect.signature(func)
        if len(args) != 0:
            tau_m = [args[i] for i, param in enumerate(signature.parameters)
                     if param == 'tau_m'][0]
            tau_s = [args[i] for i, param in enumerate(signature.parameters)
                     if param == 'tau_s'][0]
        else:
            tau_m = kwargs['tau_m']
            tau_s = kwargs['tau_s']
        check_for_valid_k_in_fast_synaptic_regime(tau_m, tau_s)
        return func(*args, **kwargs)
    return decorator_check


def pint_append(array, quantity, axis=0):
    """
    Append quantity to np.array quantity. Handles units correctly.
    
    Parameters:
    -----------
    array: pint Quantity with np.array magnitude or just np.array
        Array to which quantity should be appended.
    quantity: pint Quantity or just something unitless
        Quantity which should be appended to array.
    axis: num
        Axis along which to append quantity to array.
        
    Returns:
    --------
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
    
    quantity_list: list
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
    Create quantity with magnitude np.array with one more dimension.
    than quantity. Handles units correctly.
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


def build_full_arg_list(signature, args, kwargs):
    """
    Creates a full list of arguments including standard arguments.
    
    Parameters:
    -----------
    signature: Signature object
        The signature of a given function.
    args: list
        List of passed positional arguments.
    kwargs: dict
        Dict of passed keyword arguments.
    
    Returns:
    --------
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
