# -*- coding: utf-8 -*-
'''
Collection of routines for input and output related tasks.

Contains functions needed for reading in parameters from yaml files and
functions for saving and loading dictionaries or whole models to h5 files.

HDF5 Wrapper
************

.. autosummary::
    :toctree: _toctree/input_output/

    save_h5
    load_h5

Conversions
***********

.. autosummary::
    :toctree: _toctree/input_output/

    val_unit_to_quantities
    quantities_to_val_unit
    convert_arrays_in_dict_to_lists

Saving
******

.. autosummary::
    :toctree: _toctree/input_output/

    save_quantity_dict_to_yaml
    save_quantity_dict_to_h5
    save_network

Loading
*******

.. autosummary::
    :toctree: _toctree/input_output/

    load_val_unit_dict_from_yaml
    load_val_unit_dict_from_h5
    load_network
    load_unit_yaml

Others
******

.. autosummary::
    :toctree: _toctree/input_output/

    create_hash

'''


from __future__ import print_function
from collections.abc import Iterable
from numbers import Number
import warnings
import copy
import re

import numpy as np
import yaml
import hashlib as hl
import h5py

from . import ureg


def save_h5(file, d, *args, **kwargs):
    """
    Saves dictionary to h5 file.

    Parameters
    ----------
    file : str
        Output filename.
    d : dict
        Dictionary to be stored.
    """
    f = h5py.File(file, "w")
    try:
        _store_dict(f, d)
    finally:
        f.close()


def _store_dict(f, d):
    """
    Recursively stores dictionary in HDF5 file object.

    Parameters
    ----------
    f : HDF5 file object
        Object dictionary is to be stored in.
    d : dict
        Dictionary to be stored.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            grp = f.create_group(key)
            _store_dict(grp, value)
        else:
            if isinstance(key, Number):
                numerical_key = True
                key = str(key)
            else:
                numerical_key = False
            if isinstance(value, np.ndarray):
                dtype = value.dtype.name
            elif isinstance(value, list):
                dtypes = [type(e) for e in value]
                if len(set(dtypes)) > 1:
                    raise ValueError('List to be stored must only contain '
                                     'same datatypes.')
                elif len(value) == 0:
                    pass
                else:
                    dtype = dtypes[0]
            else:
                dtype = type(value)

            if isinstance(value, str):
                dset = f.create_dataset(key, (1,), dtype=h5py.string_dtype())
                dset[0] = value
            elif (dtype == str) or (dtype == 'str32') or (dtype == 'str64') or (dtype == 'str96'):
                dset = f.create_dataset(key, (len(value),),
                                        dtype=h5py.string_dtype())
                dset[:] = value
            else:
                value = np.array(value)
                dset = f.create_dataset(key, data=value)
            if numerical_key:
                dset.attrs['numerical_key'] = True


def load_h5(file, *args, **kwargs):
    """
    Loads dictionary from h5 file.

    Parameters
    ----------
    file : str
        File to be loaded.

    Returns
    -------
    dict
        Stored dictionary.
    """
    f = h5py.File(file, 'r')
    try:
        d = _retrieve_dict(f)
    finally:
        f.close()
    return d


def _retrieve_dict(f):
    """
    Recursively retrieves a dictionary from an HDF5 file object.

    Parameters
    ----------
    f : HDF5 file object
        Object dictionary is stored in.

    Returns
    -------
    dict
        Stored dictionary.
    """
    d = {}
    for key, group in f.items():
        # if key originally was a numerical key, convert it to number
        if group.attrs.get('numerical_key'):
            try:
                key = int(key)
            except ValueError:
                key = float(key)
        # if group is Group, retrieve recursively
        if isinstance(group, h5py._hl.group.Group):
            d[key] = _retrieve_dict(group)
        # decode bytes to strings
        elif isinstance(group[()], bytes):
            d[key] = group[()].decode('utf8')
        # convert h5py strings to python strings if necessary
        elif ((group[()].dtype == h5py.string_dtype())
              and (len(group[()]) == 1)):
            try:
                d[key] = group.asstr()[()][0]
            except AttributeError:
                d[key] = str(group[()][0])
        # convert h5py string arrays so lists of strings
        elif (group[()].dtype == h5py.string_dtype()) and (len(group[()]) > 1):
            d[key] = group.asstr()[()].tolist()
        # decode arrays of bytes to strings
        elif ((isinstance(group[()], Iterable))
              and (isinstance(group[0], bytes))):
            d[key] = np.char.decode(group[()], 'utf8')
        else:
            d[key] = group[()]
    return d


def convert_arrays_in_dict_to_lists(adict):
    """
    Recursively searches through a dict and replaces all numpy arrays by lists.

    Parameters
    ----------
    adict : dict
        Dictionary to be converted.
    """
    converted = copy.deepcopy(adict)
    for key, value in converted.items():
        if isinstance(value, dict):
            converted[key] = convert_arrays_in_dict_to_lists(value)
        elif isinstance(value, np.ndarray):
            converted[key] = value.tolist()
    return converted


def val_unit_to_quantities(dict_of_val_unit_dicts):
    """
    Recursively convert a dict of value-unit pairs to a dict of quantities.

    Combine value and unit of each quantity and save them in a dictionary
    of the structure: ``{'<quantity_key1>':<quantity1>, ...}``.

    Lists are converted to numpy arrays and then converted to quantities.

    Quantities or names without units, are just stored the way they are.

    Parameters
    ----------
    dict_of_val_unit_dicts : dict
        Dictionary of the following format::

            {'<quantity_key1>':{'val':<value1>, 'unit':<unit1>},
            '<quantity_key2>':<value2>, ...}

    Returns
    -------
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
    of the structure: ``{'<parameter1>:{'val':<value>, 'unit':<unit>}, ...}``

    Lists of quantities are handled seperately. Anything else but quantities,
    is stored just the way it is given.

    Parameters
    ----------
    dict_of_quantities : dict
        Dictionary containing only quantities (pint package) of the following
        format::

          {'<quantity_key1>':<quantity1>, ...}

    Returns
    -------
    dict
        Converted dictionary
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


def save_quantity_dict_to_yaml(file, qdict):
    """
    Convert and save dictionary of quantities to yaml file.

    Converts dict of quantities to val unit dicts and saves them in yaml file.

    Parameters
    ----------
    file : str
        Name of file.
    qdict : dict
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

    Parameters
    ----------
    file : str
        String specifying path to yaml file containing parameters in the
        following format

        .. code-block:: yaml

            <parameter1>:
                val: <value1>
                unit: <unit1>
            <parameter2>: <value_without_unit>
            ...

    Returns
    -------
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


def save_quantity_dict_to_h5(file, qdict):
    """
    Convert and save dict of quantities to h5 file.

    The quantity dictionary is first converted to a val unit dictionary and
    then saved to an h5 file.

    Parameters
    ----------
    file : str
        String specifying output file name.
    qdict : dict
        Dictionary containing quantities.
    """
    # convert data into format usable in h5 file
    output = quantities_to_val_unit(qdict)
    # save output
    try:
        save_h5(file, output)
    except KeyError:
        raise IOError(f'{file} already exists!')


def load_val_unit_dict_from_h5(file):
    """
    Load and convert val unit dict from h5 file to dict of quantities.

    The val unit dictionary is loaded from the h5 file and then converted to
    a dictionary containing quantities.

    Parameters
    ----------
    file : str
        String specifying input file name.
    """
    try:
        loaded = load_h5(file)
    except OSError:
        raise IOError(f'{file} does not exist!')

    converted = val_unit_to_quantities(loaded)
    return converted

def load_val_unit_dict(file):
    """
    Load and convert val unit dict from either h5 or yaml file to dict of
    quantities.

    Parameters
    ----------
    file : str
        String specifying input file name.
    """
    if '.yaml' in file:
        quantity_dict = load_val_unit_dict_from_yaml(file)
    elif '.h5' in file:
        quantity_dict = load_val_unit_dict_from_h5(file)
    return quantity_dict


def save_network(file, network):
    """
    Save network to h5 file.

    The networks' dictionaires (``network_params``, ``analysis_params``,
    ``results``, ``results_hash_dict``) are stored. Quantities are converted to
    value-unit dictionaries.

    Parameters
    ----------
    file : str
        Output file name.
    network : Network object
        The network to be saved.
    """
    output = {'network_params': network.network_params,
              'analysis_params': network.analysis_params,
              'results': network.results,
              'results_hash_dict': network.results_hash_dict}
    save_quantity_dict_to_h5(file, output)


def load_network(file):
    """
    Load network from h5 file.

    Parameters
    ----------
    file : str
        Input file name.

    Returns
    -------
    network_params : dict
        Network parameters.
    analysis_params : dict
        Analysis parameters.
    results : dict
        Dictionary containing most recently calculated results.
    results_hash_dict : dict
        Dictionary where all calculated results are stored.
    """
    input = load_val_unit_dict_from_h5(file)

    return (input['network_params'],
            input['analysis_params'],
            input['results'],
            input['results_hash_dict'])


def create_hash(params, param_keys):
    """
    Create unique hash from values of parameters specified in param_keys.

    Parameters
    ----------
    params : dict
        Dictionary containing all network parameters.
    param_keys : list
        List specifying which parameters should be reflected in hash.

    Returns
    -------
    str
        Hash string.
    """

    label = ''
    # add all param values to one string
    for key in sorted(list(param_keys)):
        label += str(params[key])
    # create and return hash (label must be encoded)
    return hl.md5(label.encode('utf-8')).hexdigest()


def load_unit_yaml(file):
    """
    Loads the standard unit yaml file.

    Parameters
    ----------
    file : str
        The file to be loaded.

    Returns
    -------
    dict
    """
    # try to load yaml file
    with open(file, 'r') as stream:
        try:
            unit_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return unit_dict
