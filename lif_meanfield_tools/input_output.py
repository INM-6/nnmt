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
from collections.abc import Iterable
import warnings
import copy

import numpy as np
import yaml
import hashlib as hl
import h5py_wrapper.wrapper as h5

from . import ureg


def val_unit_to_quantities(dict_of_val_unit_dicts):
    """
    Recursively convert a dict of value-unit pairs to a dict of quantities.

    Combine value and unit of each quantity and save them in a dictionary
    of the structure: {'<quantity_key1>':<quantity1>, ...}.

    Lists are converted to numpy arrays and then converted to quantities.

    Quantities or names without units, are just stored the way they are.

    Parameters:
    -----------
    dict_of_val_unit_dicts: dict
        dictionary of format {'<quantity_key1>':{'val':<value1>,
                                                 'unit':<unit1>},
                              '<quantity_key2>':<value2>,
                                                 ...}

    Returns:
    --------
    dict
        Converted dictionary of format explained above.
    """
    def formatval(val):
        """ If argument is of type list, convert to np.array. """
        if isinstance(val, list):
            return np.array(val)
        else:
            return val

    converted_dict = {}
    for key, value in dict_of_val_unit_dicts.items():
        if isinstance(value, dict):
            # if dictionary with keys val and unit, convert to quantity
            if set(('val', 'unit')) == value.keys():
                converted_dict[key] = (formatval(value['val'])
                                       * ureg.parse_expression(value['unit']))
            # if dictionary with only val, convert
            elif 'val' in value.keys():
                converted_dict[key] = formatval(value['val'])
            # if not val unit dict, first convert the dictionary
            else:
                converted_dict[key] = val_unit_to_quantities(value)
        # if not dict, convert value itself
        else:
            converted_dict[key] = formatval(value)
    return converted_dict


def quantities_to_val_unit(dict_of_quantities):
    """
    Recursively convert a dict of quantities to a dict of val-unit pairs.

    Split up value and unit of each quantiy and save them in a dictionary
    of the structure: {'<parameter1>:{'val':<value>, 'unit':<unit>}, ...}

    Lists of quantities are handled seperately. Anything else but quantities,
    is stored just the way it is given.

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
        # quantities are converted to val unit dictionary
        if isinstance(quantity, ureg.Quantity):
            converted_dict[quantity_key]['val'] = quantity.magnitude
            converted_dict[quantity_key]['unit'] = str(quantity.units)
        # nested dictionaries need to be converted first
        elif isinstance(quantity, dict):
            converted_dict[quantity_key] = quantities_to_val_unit(quantity)
        # arrays, lists and lists of quantities
        elif isinstance(quantity, Iterable):
            # lists of quantities
            if any(isinstance(part, ureg.Quantity) for part in quantity):
                converted_dict[quantity_key]['val'] = (
                    [array.magnitude for array in quantity])
                converted_dict[quantity_key]['unit'] = str(quantity[0].units)
            # arrays, lists, etc.
            else:
                converted_dict[quantity_key] = quantity
        # anything else is stored the way it is
        else:
            converted_dict[quantity_key] = quantity
    return converted_dict


def convert_arrays_in_dict_to_lists(adict):
    """
    Recursively searches through a dict and replaces all numpy arrays by lists.
    """
    converted = copy.deepcopy(adict)
    for key, value in converted.items():
        if isinstance(value, dict):
            converted[key] = convert_arrays_in_dict_to_lists(value)
        elif isinstance(value, np.ndarray):
            converted[key] = value.tolist()
    return converted


def save_quantity_dict_to_yaml(file, qdict):
    """
    Convert and save dictionary of quantities to yaml file.
    
    Converts dict of quantities to val unit dicts and saves them in yaml file.
    
    Parameters:
    -----------
    file: str
        Name of file.
    qdict: dict
        Dictionary containing quantities.
    """
    converted = quantities_to_val_unit(qdict)
    converted = convert_arrays_in_dict_to_lists(converted)
    with open(file, 'w') as f:
        yaml.dump(converted, f)
    

def load_val_unit_dict_from_yaml(file):
    """
    Load and convert val unit dictionary from yaml file.

    Load val unit dictionary from yaml file and convert it to dictionary of
    quantities.

    Parameters:
    -----------
    file : str
        string specifying path to yaml file containing parameters in format
        <parameter1>:
            val: <value1>
            unit: <unit1>
        <parameter2>: <value_without_unit>
        ...

    Returns:
    --------
    dict
        dictionary containing all converted val unit dicts as quantities
    """
    # try to load yaml file
    with open(file, 'r') as stream:
        try:
            val_unit_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    # convert parameters to quantities
    quantity_dict = val_unit_to_quantities(val_unit_dict)
    # return converted network parameters
    return quantity_dict


def create_hash(params, param_keys):
    """
    Create unique hash from values of parameters specified in param_keys.

    Parameters:
    -----------
    params : dict
        Dictionary containing all network parameters.
    param_keys : list
        List specifying which parameters should be reflected in hash.

    Returns:
    --------
    str
        Hash string.
    """

    label = ''
    # add all param values to one string
    for key in sorted(list(param_keys)):
        label += str(params[key])
    # create and return hash (label must be encoded)
    return hl.md5(label.encode('utf-8')).hexdigest()
    
    
def save_network(file, network, overwrite=False):
    """
    Save network to h5 file.
    
    The networks' dictionaires (network_params, analysis_params, results,
    results_hash_dict) are stored. Quantities are converted to value-unit
    dictionaries.
    
    Parameters:
    -----------
    file: str
        Output file name.
    network: Network object
        The network to be saved.
    overwrite: bool
        Whether to overwrite an existing h5 file or not. If there already is
        one, h5py tries to update the h5 dictionary.
    """
    network_params = quantities_to_val_unit(network.network_params)
    analysis_params = quantities_to_val_unit(network.analysis_params)
    results = quantities_to_val_unit(network.results)
    results_hash_dict = quantities_to_val_unit(network.results_hash_dict)
    
    output = {'network_params': network_params,
              'analysis_params': analysis_params,
              'results': results,
              'results_hash_dict': results_hash_dict}
    try:
        h5.save(file, output, overwrite_dataset=overwrite)
    except KeyError:
        raise IOError(f'{file} already exists! Use `overwrite=True` if you '
                      'want to overwrite it.')
    
    
def load_network(file):
    """
    Load network from h5 file.
    
    Parameters:
    -----------
    file: str
        Input file name.
    
    Returns:
    --------
    network_params: dict
        Network parameters.
    analysis_params: dict
        Analysis parameters.
    results: dict
        Dictionary containing most recently calculated results.
    results_hash_dict: dict
        Dictionary where all calculated results are stored.
    """
    try:
        input = h5.load(file)
    # if not existing OSError is raised by h5py_wrapper, then return empty dict
    except OSError:
        message = f'File {file} not found!'
        warnings.warn(message)
        return {}, {}, {}, {}
    
    network_params = val_unit_to_quantities(input['network_params'])
    analysis_params = val_unit_to_quantities(input['analysis_params'])
    results = val_unit_to_quantities(input['results'])
    results_hash_dict = val_unit_to_quantities(input['results_hash_dict'])
    
    return network_params, analysis_params, results, results_hash_dict


def save_quantity_dict_to_h5(file, qdict, overwrite=False):
    """
    Convert and save dict of quantities to h5 file.
    
    The quantity dictionary is first converted to a val unit dictionary and
    then saved to an h5 file.

    Parameters:
    -----------
    file: str
        String specifying output file name.
    qdict: dict
        Dictionary containing quantities.
    overwrite: bool
        Whether h5 file should be overwritten, if already existing.
    """
    # convert data into format usable in h5 file
    output = quantities_to_val_unit(qdict)
    # save output
    try:
        h5.save(file, output, overwrite_dataset=overwrite)
    except KeyError:
        raise IOError(f'{file} already exists! Use `overwrite=True` if you '
                      'want to overwrite it.')


def load_from_h5(network_params={}, param_keys=[], file=''):
    """
    Load existing results and analysis_params for given params from h5 file.

    Loads results from h5 files named with the standard format
    <label>_<hash>.h5, if this file already exists. Or uses given list of
    parameters to create hash to find file. Or reads from file specified in
    file.

    Parameters:
    -----------
    network_params : dict
        Dictionary containing network parameters as quantities.
    param_keys: list
        List of parameters used in file hash.
    file: str
        optional string specifying input file name
        (default: <label>_<hash>.h5).

    Returns:
    --------
    analysis_params: dict
        Dictionary containing all found analysis_params.
    results: dict
        Dictionary containing all found results.
    """
    # if no input file name is specified
    if not file:
        # create hash from given parameters
        # collect all parameters reflected by hash in one dictionary
        hash_params = {}
        hash_params.update(network_params)
        # if user did not specify which parameters to use for hash
        if not param_keys:
            # take all parameters sorted alphabetically
            param_keys = sorted(list(hash_params.keys()))
        # crate hash from param_keys
        hash = create_hash(hash_params, param_keys)
        # default input name
        file = '{}_{}.h5'.format(network_params['label'], str(hash))

    # try to load file with standard name
    try:
        input_file = h5.load(file)
    # if not existing OSError is raised by h5py_wrapper, then return empty dict
    except OSError:
        return {}, {}

    # read in whats already stored
    analysis_params = input_file['analysis_params']
    results = input_file['results']

    # convert results to quantitites
    analysis_params = val_unit_to_quantities(analysis_params)
    results = val_unit_to_quantities(results)

    return analysis_params, results


def load_h5(filename):
    """
    filename: str
        default filename format is ''<label>_<hash>.h5'
    """
    try:
        raw_data = h5.load(filename)
    except OSError:
        raise IOError(f'{filename} does not exist!')

    data = {}
    for key in sorted(raw_data.keys()):
        data[key] = val_unit_to_quantities(raw_data[key])
    return data
