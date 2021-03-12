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

import numpy as np
import yaml
import hashlib as hl
import h5py_wrapper.wrapper as h5

from . import ureg


def val_unit_to_quantities(dict_of_val_unit_dicts):
    """
    Convert a dictionary of value-unit pairs to a dictionary of quantities

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
        # if dictionary with keys val and unit, convert to quantity
        if isinstance(value, dict) and set(('val', 'unit')) == value.keys():
            converted_dict[key] = (formatval(value['val'])
                                   * ureg.parse_expression(value['unit']))
        else:
            converted_dict[key] = formatval(value)
    return converted_dict


def quantities_to_val_unit(dict_of_quantities):
    """
    Convert a dictionary of quantities to a dictionary of val-unit pairs

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
        # lists of strings need to be treated seperately
        elif isinstance(quantity, Iterable):
            if any(isinstance(part, str) for part in quantity):
                converted_dict[quantity_key] = quantity
            elif any(isinstance(part, ureg.Quantity) for part in quantity):
                converted_dict[quantity_key]['val'] = (
                    [array.magnitude for array in quantity])
                converted_dict[quantity_key]['unit'] = str(quantity[0].units)
            else:
                converted_dict[quantity_key] = quantity
        # anything else is stored the way it is
        else:
            converted_dict[quantity_key] = quantity
    return converted_dict


def load_params(file_path):
    """
    Load and convert parameters from yaml file

    Load parameters from yaml file and convert them from value unit
    dictionaries (used in yaml file) to quantities (used in implementation of
    functions in meanfield_calcs.py).

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
    
    
def convert_results_hash_dict_quantities_to_val_unit(hash_dict):
    for hash, result in hash_dict.items():
        if 'analysis_params' in result.keys():
            result['analysis_params'] = quantities_to_val_unit(
                result['analysis_params'])
        hash_dict[hash] = quantities_to_val_unit(result)
    return hash_dict


def convert_results_hash_dict_val_unit_to_quantities(hash_dict):
    for hash, result in hash_dict.items():
        result = val_unit_to_quantities(result)
        if 'analysis_params' in result.keys():
            result['analysis_params'] = val_unit_to_quantities(
                result['analysis_params'])
        hash_dict[hash] = result
    return hash_dict

    
def save_network(file_name, network, overwrite_dataset=False):
    """
    Save network to h5 file.
    
    The networks' dictionaires (network_params, analysis_params, results,
    results_hash_dict) are stored. Quantities are converted to value-unit
    dictionaries.
    
    Parameters:
    -----------
    file_name: str
        Output file name.
    network: Network object
        The network to be saved.
    overwrite_dataset: bool
        Whether to overwrite an existing h5 file or not. If there already is
        one h5py tries to update the h5 dictionary.
    """
    network_params = quantities_to_val_unit(network.network_params)
    analysis_params = quantities_to_val_unit(network.analysis_params)
    results = quantities_to_val_unit(network.results)
    results_hash_dict = convert_results_hash_dict_quantities_to_val_unit(
        network.results_hash_dict)
    
    output = {'network_params': network_params,
              'analysis_params': analysis_params,
              'results': results,
              'results_hash_dict': results_hash_dict}
    h5.save(file_name, output, overwrite_dataset=overwrite_dataset)
    
    
def load_network(file_name):
    """
    Load network from h5 file.
    
    Parameters:
    -----------
    file_name: str
        Output file name.
    
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
        input = h5.load(file_name)
    # if not existing OSError is raised by h5py_wrapper, then return empty dict
    except OSError:
        print(f'File {file_name} not found!')
        return {}, {}, {}, {}
    
    network_params = val_unit_to_quantities(input['network_params'])
    analysis_params = val_unit_to_quantities(input['analysis_params'])
    results = val_unit_to_quantities(input['results'])
    results_hash_dict = convert_results_hash_dict_val_unit_to_quantities(
        input['results_hash_dict'])
    
    return network_params, analysis_params, results, results_hash_dict


def save_params(output_key, output, file_name):
    """
    Save data and given parameters in h5 file.

    By default the output name will be <label>_<hash>.h5, where the hash is
    created using network_params. But you can either specify an ouput_name
    yourself, or specify which param_keys should be reflected in the hash.

    Parameters:
    -----------
    results_dict : dict
        Dictionary containing all calculated results.
    network_params : dict
        Dictionary containing network parameters as quantities.
    analysis_params: dict
        Dictionary containing analysis parameters as quantities.
    file_name: str
        String specifying output file name.
    """
    # convert data into format usable in h5 file
    output = quantities_to_val_unit(output)
    output_dict = {}
    output_dict[output_key] = output

    # save output
    h5.save(file_name, output_dict, overwrite_dataset=True)


def load_from_h5(network_params={}, param_keys=[], input_name=''):
    """
    Load existing results and analysis_params for given params from h5 file.

    Loads results from h5 files named with the standard format
    <label>_<hash>.h5, if this file already exists. Or uses given list of
    parameters to create hash to find file. Or reads from file specified in
    input_name.

    Parameters:
    -----------
    network_params : dict
        Dictionary containing network parameters as quantities.
    param_keys: list
        List of parameters used in file hash.
    input_name: str
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
    if not input_name:
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
        input_name = '{}_{}.h5'.format(network_params['label'], str(hash))

    # try to load file with standard name
    try:
        input_file = h5.load(input_name)
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
        raw_data = {}

    data = {}
    for key in sorted(raw_data.keys()):
        data[key] = val_unit_to_quantities(raw_data[key])
    return data
