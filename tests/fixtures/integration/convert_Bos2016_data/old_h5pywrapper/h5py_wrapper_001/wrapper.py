import re
import h5py
if int(re.sub('\.', '', h5py.version.version)) < 230:
    raise ImportError(
        "Using h5py version %s. Version must be >= 2.3.0" % (h5py.version.version))

import numpy
import collections
import quantities as pq
from subprocess import call
import ast

######################################################################
# Wrapper to conveniently store arbitrarily nested python dictionaries
# to HDF5 files.  There is an outdated version and a new version:
# a)Outdated: The dictionaries were flattened by joining the keys of
#   different levels of the dictionary and then stored as datasets to a
#   HDF5 file
# b)New: The dictionaries are stored in an HDF5 file by
#   creating groups for every level and a dataset for the value in the
#   lowest level
# There is a transform function which simply takes a file created in
# the outdated manner and converts it to a file of the new kind.
# There is a function storing and loading an example dictionary.
# IMPORTANT NOTE:
# h5py uses numpy.arrays to load datasets since this enables users to
# load only parts of a dataset. this means all lists will be converted
# to arrays when they are loaded from an h5 file. currently there is
# no option to change this behaviour. you need to do this manually
# after loading the file.
# (see also http://alfven.org/wp/2011/11/psa-why-using-dataset-value-is-discouraged-in-h5py/)
######################################################################

######################################################################
# a) Outdated version


def load_h5_old(filename, sep='_'):
    '''
    Loads h5-file and extracts the dictionary within it.

    Outputs:
      dict - dictionary, one or several pairs of string and any type of variable,
             e.g dict = {'name1': var1,'name2': var2}
    '''

    f = h5py.File(filename, 'r')
    flat_dict = {}
    for k, v in list(f.items()):
        value = numpy.array(v[:])
        value = list(v[:])

        if len(value):
            flat_dict[k] = value[0]
        else:
            flat_dict[k] = value
    f.close()
    dic = unflatten(flat_dict, separator=sep)
    return dic


# builds a nested dictionary out of a flattened dictionary
def unflatten(dictionary, separator='_'):
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(separator)
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict

######################################################################
# b) New Version

# Auxiliary functions

def delete_group(f, group):
    try:
        f = h5py.File(f, 'r+')
        try:
            del f[group]
            f.close()
        except KeyError:
            f.close()
    except IOError:
        pass


def node_exists(f, key):
    f = h5py.File(f, 'r')
    exist = key in f
    f.close()
    return exist


def dict_to_h5(d, f, overwrite_dataset, compression=None, **keywords):
    if 'parent_group' in keywords:
        parent_group = keywords['parent_group']
    else:
        parent_group = f.parent

    for k, v in list(d.items()):
        if isinstance(v, collections.MutableMapping):
            if parent_group.name != '/':
                group_name = parent_group.name + '/' + str(k)
            else:
                group_name = parent_group.name + str(k)
            group = f.require_group(group_name)
            dict_to_h5(
                v, f, overwrite_dataset, parent_group=group, compression=compression)
        else:
            if not str(k) in list(parent_group.keys()):
                create_dataset(parent_group, k, v, compression=compression)
            else:
                if overwrite_dataset == True:
                    del parent_group[str(k)]  # delete the dataset
                    create_dataset(parent_group, k, v, compression=compression)
                else:
                    print('Dataset', str(k), 'already exists!')
    return 0  # ?


def create_dataset(parent_group, k, v, compression=None):
    shp = numpy.shape(v)
    if v is None:
        parent_group.create_dataset(
            str(k), data='None', compression=compression)
    else:
        if isinstance(v, (list, numpy.ndarray)):
            if numpy.array(v).dtype.name == 'object':
                if len(shp) > 1:
                    print('Dataset', k, 'has an unsupported format!')
                else:
                    # store 2d array with an unsupported format by reducing
                    # it to a 1d array and storing the original shape
                    # this does not work in 3d!
                    oldshape = numpy.array([len(x) for x in v])
                    data_reshaped = numpy.hstack(v)
                    data_set = parent_group.create_dataset(
                        str(k), data=data_reshaped, compression=compression)
                    data_set.attrs['oldshape'] = oldshape
                    data_set.attrs['custom_shape'] = True
            elif isinstance(v, pq.Quantity):
                data_set = parent_group.create_dataset(str(k), data=v)
                data_set.attrs['_unit'] = v.dimensionality.string
            else:
                data_set = parent_group.create_dataset(
                    str(k), data=v, compression=compression)
        # ## ignore compression argument for scalar datasets
        elif isinstance(v, (int, float)):
            data_set = parent_group.create_dataset(str(k), data=v)
        else:
            data_set = parent_group.create_dataset(
                str(k), data=v, compression=compression)

        # ## Explicitely store type of key
        _key_type = type(k).__name__
        data_set.attrs['_key_type'] = _key_type


def dict_from_h5(f, lazy=False):
    # .value converts everything(?) to numpy.arrays
    # maybe there is a different option to load it, to keep, e.g., list-type
    if h5py.h5i.get_type(f.id) == 5:  # check if f is a dataset
        if not lazy:
            if hasattr(f, 'value'):
                # ## This if-branch exists to enable loading of deprecated hdf5 files
                if 'EMPTYARRAY' in str(f.value):
                    shp = f.value.split('_')[1]
                    shp = tuple(int(i)
                                for i in shp[1:-1].split(',') if i != '')
                    return numpy.reshape(numpy.array([]), shp)
                elif str(f.value) == 'None':
                    return None
                else:
                    if len(list(f.attrs.keys())) > 0 and 'custom_shape' in list(f.attrs.keys()):
                        if f.attrs['custom_shape']:
                            return load_custom_shape(f.attrs['oldshape'], f.value)
                    else:
                        return f.value
            else:
                return numpy.array([])
        else:
            return None
    else:
        d = {}
        items = list(f.items())
        for name, obj in items:
            # Check if obj is a group or a dataset
            if h5py.h5i.get_type(obj.id) == 2:
                sub_d = dict_from_h5(obj, lazy=lazy)
                d[name] = sub_d
            else:
                if not lazy:
                    if hasattr(obj, 'value'):
                        if 'EMPTYARRAY' in str(obj.value):
                            shp = obj.value.split('_')[1]
                            shp = tuple(int(i)
                                        for i in shp[1:-1].split(',') if i != '')
                            d[name] = numpy.reshape(numpy.array([]), shp)
                        elif str(obj.value) == 'None':
                            d[name] = None
                        else:
                            # if dataset has custom_shape=True, we rebuild the
                            # original array
                            if len(list(obj.attrs.keys())) > 0:
                                if 'custom_shape' in list(obj.attrs.keys()):
                                    if obj.attrs['custom_shape']:
                                        d[name] = load_custom_shape(
                                            obj.attrs['oldshape'], obj.value)
                                elif '_unit' in list(obj.attrs.keys()):
                                    d[name] = pq.Quantity(
                                        obj.value, obj.attrs['_unit'])
                                elif '_key_type' in list(obj.attrs.keys()):
                                    # added string_ to handle numpy.string_,
                                    # TODO: find general soluation for numpy
                                    # data types
                                    if obj.attrs['_key_type'] not in ['str', 'unicode', 'string_']:
                                        d[ast.literal_eval(name)] = obj.value
                                    else:
                                        d[name] = obj.value
                            else:
                                d[name] = obj.value
                    else:
                        d[name] = numpy.array([])
                else:
                    d[name] = None
        return d


def load_custom_shape(oldshape, oldata):
    data_reshaped = []
    counter = 0
    for l in oldshape:
        data_reshaped.append(numpy.array(oldata[counter:counter + l]))
        counter += l
    return numpy.array(data_reshaped, dtype=object)


# Save routine
def add_to_h5(filename, d, write_mode='a', overwrite_dataset=False, resize=False, dict_label='', compression=None):
    '''
    Save dictionary containing data to hdf5 file.

    **Args**:
        filename: file name of the hdf5 file to be created
        d: dictionary to be stored
        write_mode: can be set to 'a'(append) or 'w'(overwrite), analog to normal file handling in python (default='a')
        overwrite_dataset: whether all datasets should be overwritten if already existing. (default=False)
        resize: if True, the hdf5 file is resized after writing all data, may reduce file size, caution: slows down writing (default=False)
        dict_label: If given, the dictionary is stored as a group with the given name in the hdf5 file, labels can also given as paths to target lower levels of groups, e.g.: dict_label='test/trial/spiketrains' (default='')
        compression: Compression strategy to reduce file size.  Legal values are 'gzip', 'szip','lzf'.  Can also use an integer in range(10) indicating gzip, indicating the level of compression. 'gzip' is recommended. Caution: This slows down writing and loading of data. Attention: Will be ignored for scalar data.

    '''
    try:
        f = h5py.File(filename, write_mode)
    except IOError:
        raise IOError(
            'unable to create ' + filename + ' (File accessability: Unable to open file)')
    if dict_label != '':
        base = f.require_group(dict_label)
        dict_to_h5(
            d, f, overwrite_dataset, parent_group=base, compression=compression)
    else:
        dict_to_h5(d, f, overwrite_dataset, compression=compression)
    fname = f.filename
    f.close()
    if overwrite_dataset == True and resize == True:
        call(['h5repack', '-i', fname, '-o', fname + '_repack'])
        call(['mv', fname + '_repack', fname])
    return 0

# Load routine


def load_h5(filename, path='', lazy=False):
    '''
    The Function returns a dictionary of all dictionaries that are
    stored in the HDF5 File.

    **Args**:
        filename: file name of the hdf5 file to be loaded
        path: argument to access deeper levels in the hdf5 file (default='')
        lazy: boolean, default: False
              If True, only the structure of the file is loaded without actual data    
    '''

    d = {}
    try:
        f = h5py.File(filename, 'r')
    except IOError:
        raise IOError('unable to open \"' + filename +
                      '\" (File accessability: Unable to open file)')
    if path == '':
        d = dict_from_h5(f, lazy=lazy)
    else:
        if path[0] == '/':
            path = path[1:]
        if node_exists(filename, path):
            d = dict_from_h5(f[path], lazy=lazy)
        else:
            f.close()
            raise KeyError('unable to open \"' + filename + '/' +
                           path + '\" (Key accessability: Unable to access key)')
    f.close()
    return d


######################################################################
# Transform outdated file to new file
def transform_h5(filename, new_filename):
    '''
    Transform function which simply takes a file created in
    the outdated manner and converts it to a file of the new kind.
    '''
    x = load_h5_old(filename)
    add_to_h5(new_filename, x)


######################################################################

def example():
    filename = 'example.hdf5'
    d = {}
    d['a'] = {'a1': numpy.array([1, 2, 3]),
              'a2': 4.0,
              'a3': {'a31': 'Test'}}
    d['b'] = numpy.arange(0., 0.5, 0.01)
    d['c'] = 'string'

    # # save dictionary to file
    add_to_h5(filename, d)

    # ## load dictionary from file
    dd = load_h5(filename)

    print(dd)

    return 0
