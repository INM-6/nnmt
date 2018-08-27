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

# read arguments from shell command
args = docopt.docopt(__doc__)

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

def create_hash(network_params):
    """
    Create unique hash from given network parameters

    Parameters:
    -----------
    network_params : dict
        dictionary containing all network parameters

    Returns:
    --------
    str
        hash string
    """

def save(data_dict, network_params, output_name=''):
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
