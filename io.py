#!/usr/bin/env python
# encoding:utf8
'''
Handles reading-in yaml files and converting the physical parameters, specified
in yaml files, to theoretical parameters, needed for usage of given implemented
functions (they rely on a redefinition of quantities). Handles output-writing
and provides function for creating hashes to uniquely identify output files.

Usage: io.py [options]

Options:
    -h, --help       show extensive usage information
'''

from __future__ import print_function

import docopt
import pprint

import numpy as np

import yaml
from pint import UnitRegistry
import hashlib as hl
import h5py_wrapper.wrapper as h5

ureg = UnitRegistry()

def val_unit_to_quantities(dict_of_val_unit_dicts):
    """
    Convert a dictionary of value-unit pairs to a dictionary of quantities

    Combine value and unit of each quantity and save them in a dictionary
    of the structure: {'<quantity_key1>':<quantity1>, ...}

    Parameters:
    -----------
    dict_of_val_unit_dicts: dict
        dictionary of format {'<quantity_key1>':{'val':<value1>,
                                                 'unit':<unit1>},
                                                 ...}

    Returns:
    --------
    dict
        converted dictionary of format
    """

    converted_dict = {}
    for quantity_key, val_unit_dict in dict_of_val_unit_dicts.items():
        # if value is given as nested list, convert to numpy array
        if isinstance(val_unit_dict['val'], list):
            if any(isinstance(part, list) for part in val_unit_dict['val']):
                print(val_unit_dict['val'])
                print(np.array(val_unit_dict['val']))

        # if unit is specified, convert value unit pair to quantity
        if 'unit' in val_unit_dict:
            converted_dict[quantity_key] = (val_unit_dict['val']
            * ureg.parse_expression(val_unit_dict['unit']))
        # as strings can't be represented as a quantities,
        # they needs to be treated seperately
        elif isinstance(val_unit_dict['val'], str):
            converted_dict[quantity_key] = val_unit_dict['val']
        else:
            # check that parameters are specified in correct format
            try:
                converted_dict[quantity_key] = ureg.Quantity(val_unit_dict['val'])
            except TypeError as error:
                raise KeyError(('Check that value of parameter in {} is given '
                + 'as value belonging to key "val" (syntax: '
                + '"val: <value>")').format(file_path))

    return converted_dict


def quantities_to_val_unit(dict_of_quantities):
    """
    Convert a dictionary of quantities to a dictionary of val-unit pairs

    Split up value and unit of each quantiy and save them in a dictionary
    of the structure: {'<parameter1>:{'val':<value>, 'unit':<unit>}, ...}

    Parameters:
    -----------
    dict_containing_quantities: dict
        dictionary containing only quantities (pint package) of format
        {'<quantity_key1>':<quantity1>, ...}

    Returns:
    --------
    dict
        converted dictionary
    """

    converted_dict = {}
    for quantity_key, quantity in dict_of_quantities.items():
        converted_dict[quantity_key] = {}
        # as strings can't be represented as a quantities,
        # they needs to be treated seperately
        if isinstance(quantity, str):
            converted_dict[quantity_key] = quantity
        else:
            converted_dict[quantity_key]['val'] = quantity.magnitude
            converted_dict[quantity_key]['unit'] = str(quantity.units)

    return converted_dict


def load_params(file_path):
    """
    Load and convert parameters from yaml file

    Load parameters from yaml file and convert them from value unit dictionaries
    (used in yaml file) to quantities (used in implementation of functions in
    meanfield_calcs.py).

    Parameters:
    -----------
    file_path : str
        string specifying path to yaml file containing parameters in format
        <parameter1>:
            val: <value1>
            unit: <unit1>
        ...

    Returns:
    --------
    dict
        dictionary containing all converted parameters as quantities
    """

    # try to load yaml file
    with open(file_path, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # convert parameters to quantities
    params_converted = val_unit_to_quantities(params)

    # return converted network parameters
    return params_converted


# def load_analysis_params(file_path):
#     """
#     Load analysis paramters from yaml file
#
#     Load parameters needed to define calculations done within meanfield_calcs.py
#     like for example the minimum and maximum frequency considered, or the
#     increment width.
#
#     Parameters:
#     -----------
#     file_path : str
#         string specifying path to yaml file containing analysis paramters
#
#     Returns:
#     --------
#     dict
#         dictionary containing all parameters
#     """
#
#     # try to load yaml file
#     with open(file_path, 'r') as stream:
#         try:
#             analysis_params = yaml.safe_load(stream)
#         except yaml.YAMLError as exc:
#             print(exc)
#
#     analysis_params
#
#     return analysis_params


def create_hash(params, param_keys):
    """
    Create unique hash from values of parameters specified in param_keys

    Parameters:
    -----------
    params : dict
        dictionary containing all network parameters
    param_keys : list
        list specifying which parameters should be reflected in hash

    Returns:
    --------
    str
        hash string
    """

    label = ''
    # add all param values to one string
    for key in param_keys:
        label += str(params[key])
    # create and return hash (label must be encoded)
    return hl.md5(label.encode('utf-8')).hexdigest()


def save(data_dict, network_params, param_keys=[], output_name=''):
    """
    Save data and given paramters in h5 file

    Parameters:
    -----------
    data : dict
        dictionary containing all calculated data
    network_params: dict
        dictionary containing all network parameters
    output_name : str
        optional string specifying output file name (default: <label>_<hash>.h5)
    """

    # is user did not specify output name
    if not output_name:
        # if user did not specify which parameters to use for hash
        if not param_keys:
            # take all parameters sorted alphabetically
            param_keys = sorted(list(network_params.keys()))
        # crate hash from param_keys
        hash = create_hash(network_params, param_keys)
        # default output name
        output_name = '{}_{}.h5'.format(network_params['label'], str(hash))

    # convert data and network params into format usable in h5 files
    data = quantities_to_val_unit(data_dict)
    network_params = quantities_to_val_unit(network_params)
    output = dict(network_params=network_params, data=data)
    # save output
    h5.save(output_name, output, overwrite_dataset=True)

if __name__ == '__main__':
    params = load_network_params('network_params_microcircuit.yaml')
    print(params)
    save(params,params)
