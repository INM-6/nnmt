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

import yaml
from pint import UnitRegistry
import hashlib as hl

ureg = UnitRegistry()

def load_network_params(file_path):
    """
    Load and convert parameters from yaml file

    Load network parameters from yaml file and convert them from physical
    paramters (used in yaml file) to theoretical parameters (used in
    implementation of functions in meanfield_calcs.py).

    Parameters:
    -----------
    file_path : str
        string specifying path to yaml file containing network parameters

    Returns:
    --------
    dict
        dictionary containing all converted parameters
    """

    # try to load yaml file
    with open(file_path, 'r') as stream:
        try:
            network_params_phys = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # convert parameters to quantities
    network_params_thy = {}
    for param, quantity in network_params_phys.items():
        # if quantity is given as dictionary specifying value and unit,
        # convert them to quantity
        if 'val' in quantity and 'unit' in quantity:
            network_params_thy[param] = (quantity['val']
            * ureg.parse_expression(quantity['unit']))
        else:
            network_params_thy[param] = quantity

    # return converted network parameters
    return network_params_thy


def load_analysis_params(file_path):
    """
    Load analysis paramters from yaml file

    Load parameters needed to define calculations done within meanfield_calcs.py
    like for example the minimum and maximum frequency considered, or the
    increment width.

    Parameters:
    -----------
    file_path : str
        string specifying path to yaml file containing analysis paramters

    Returns:
    --------
    dict
        dictionary containing all parameters
    """

    # try to load yaml file
    with open(file_path, 'r') as stream:
        try:
            analysis_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return analysis_params


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
