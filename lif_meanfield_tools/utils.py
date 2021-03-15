"""
This module contains helper functions.
"""

import inspect
import warnings
from functools import wraps
import numpy as np

from . import ureg


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
    if (0.1 < k) & (k < 1):
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
