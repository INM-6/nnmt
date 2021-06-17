# -*- coding: utf-8 -*-
"""
Unit tests for the input output module.
"""

import pytest
import numpy as np
import h5py_wrapper as h5
import warnings
import yaml

import nnmt
import nnmt.input_output as io

from ...checks import (check_file_in_tmpdir,
                      check_quantity_dicts_are_equal,
                      check_dict_contains_no_quantity,
                      check_dict_contains_no_val_unit_dict,
                      assert_array_equal,
                      assert_units_equal)

ureg = nnmt.ureg

path_to_fixtures = 'tests/fixtures/unit/config/'

val_unit_pairs = [
    dict(numerical=1),
    dict(quantity={'val': 1, 'unit': 'hertz'}),
    dict(only_val_dict={'val': 1}),
    dict(string='test'),
    dict(list_of_strings=['spam', 'ham']),
    dict(array=np.array([1, 2, 3])),
    dict(quantity_array={'val': np.array([1, 2, 3]), 'unit': 'second'}),
    dict(quantity_list={'val': [1, 2, 3], 'unit': 'second'}),
    dict(two_d_array=np.arange(9).reshape(3, 3)),
    dict(two_d_quantity_array={'val': np.arange(9).reshape(3, 3),
                               'unit': 'second'}),
    dict(two_d_list_of_quantites={'val': [[i for i in range(3)]
                                          for j in range(3)],
                                  'unit': 'second'}),
    dict(numerical=1,
         quantity={'val': 1, 'unit': 'hertz'},
         string='test',
         list_of_strings=['spam', 'ham'],
         array=np.array([1, 2, 3]),
         quantity_array={'val': np.array([1, 2, 3]), 'unit': 'second'},
         quantity_list={'val': [1, 2, 3], 'unit': 'second'},
         two_d_array=np.arange(9).reshape(3, 3),
         two_d_quantity_array={'val': np.arange(9).reshape(3, 3),
                               'unit': 'second'},
         two_d_list_of_quantites={'val': [[i for i in range(3)]
                                          for j in range(3)],
                                  'unit': 'second'},
         ),
    ]

quantity_dicts = [
    dict(numerical=1),
    dict(quantity=1 * ureg.Hz),
    dict(only_val_dict=1),
    dict(string='test'),
    dict(list_of_strings=['spam', 'ham']),
    dict(array=np.array([1, 2, 3])),
    dict(quantity_array=np.array([1, 2, 3]) * ureg.s),
    dict(quantity_list=np.array([1, 2, 3]) * ureg.s),
    dict(two_d_array=np.arange(9).reshape(3, 3)),
    dict(two_d_quantity_array=np.arange(9).reshape(3, 3) * ureg.s),
    dict(two_d_list_of_quantites=[[i for i in range(3)]
                                  for j in range(3)] * ureg.s),
    dict(numerical=1,
         quantity=1 * ureg.Hz,
         string='test',
         list_of_strings=['spam', 'ham'],
         array=np.array([1, 2, 3]),
         quantity_array=np.array([1, 2, 3]) * ureg.s,
         quantity_list=np.array([1, 2, 3]) * ureg.s,
         two_d_array=np.arange(9).reshape(3, 3),
         two_d_quantity_array=np.arange(9).reshape(3, 3) * ureg.s,
         two_d_list_of_quantites=[[i for i in range(3)]
                                  for j in range(3)] * ureg.s),
    ]

ids = [list(val_unit_pair.keys())[0] for val_unit_pair in val_unit_pairs[:-1]]
ids.append('mixed')


@pytest.fixture
def param_test_dict():
    return dict(string='test',
                list_of_strings=['spam', 'ham'],
                numerical=1,
                list=[1, 2, 3],
                two_d_array=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                quantity=1 * ureg.s,
                quantity_list=np.array([1, 2, 3]) * ureg.s,
                quantity_two_d_array=np.array([[1, 2, 3],
                                               [4, 5, 6],
                                               [7, 8, 9]]) * ureg.s
                )


class Test_val_unit_to_quantities:

    @pytest.mark.parametrize('val_unit_pair, quantity_dict',
                             zip(val_unit_pairs, quantity_dicts),
                             ids=ids)
    def test_dict_of_val_unit_pairs_is_converted_to_dict_of_quantities(
            self, val_unit_pair, quantity_dict):
        converted = io.val_unit_to_quantities(val_unit_pair)
        while converted:
            conv_item = converted.popitem()
            exp_item = quantity_dict.popitem()
            try:
                # dict value has just one element
                assert conv_item == exp_item
            except ValueError:
                # dict value has more than one element
                # check key
                assert conv_item[0] == exp_item[0]
                # check value
                assert_array_equal(conv_item[1], exp_item[1])
                assert_units_equal(conv_item[1], exp_item[1])

    def test_unit_abbreviations_work_correctly(self):
        val_unit_pairs = dict(
            hertz={'val': 1, 'unit': 'Hz'},
            second={'val': 1, 'unit': 's'},
            meter={'val': 1, 'unit': 'm'},
            volt={'val': 1, 'unit': 'V'},
            ampere={'val': 1, 'unit': 'A'},
            )
        quantity_dict = dict(
            hertz=1 * ureg.Hz,
            second=1 * ureg.s,
            meter=1 * ureg.m,
            volt=1 * ureg.V,
            ampere=1 * ureg.A,
            )
        converted = io.val_unit_to_quantities(val_unit_pairs)
        assert converted == quantity_dict
        
    def test_nested_dictionaries_are_converted_correctly(self):
        test = dict(a=dict(a1=dict(val=1, unit='hertz'),
                           a2=dict(val=2, unit='ms')),
                    b=dict(b1=dict(val=1, unit='hertz'),
                           b2=dict(val=2, unit='ms')),
                    c=dict(val=3, unit='meter'))
        converted = io.val_unit_to_quantities(test)
        check_dict_contains_no_val_unit_dict(converted)
        assert isinstance(converted['a']['a1'], ureg.Quantity)
        assert isinstance(converted['a']['a2'], ureg.Quantity)
        assert isinstance(converted['b']['b1'], ureg.Quantity)
        assert isinstance(converted['b']['b2'], ureg.Quantity)
        assert isinstance(converted['c'], ureg.Quantity)


class Test_quantities_to_val_unit:

    @pytest.mark.parametrize('quantity_dict, val_unit_pair',
                             zip(quantity_dicts, val_unit_pairs),
                             ids=ids)
    def test_dict_of_quantities_is_converted_to_dict_of_val_unit_pairs(
            self, quantity_dict, val_unit_pair):
        converted = io.quantities_to_val_unit(quantity_dict)
        while converted:
            conv_item = converted.popitem()
            exp_item = val_unit_pair.popitem()
            # check key
            assert conv_item[0] == exp_item[0]
            try:
                # check dict value which is a val_unit_dict
                assert conv_item[1]['unit'] == exp_item[1]['unit']
                try:
                    assert conv_item[1]['val'] == exp_item[1]['val']
                except ValueError:
                    assert_array_equal(conv_item[1]['val'], exp_item[1]['val'])
            except (IndexError, TypeError):
                try:
                    assert conv_item[1] == conv_item[1]
                except ValueError:
                    assert_array_equal(conv_item[1], exp_item[1])

    def test_list_of_quantities(self):
        quantity_dict = dict(list_of_quantities=[1 * ureg.Hz,
                                                 2 * ureg.Hz,
                                                 3 * ureg.Hz])
        val_unit_pair = dict(list_of_quantities={'val': np.array([1, 2, 3]),
                                                 'unit': 'hertz'})
        converted = io.quantities_to_val_unit(quantity_dict)
        conv_item = converted.popitem()
        exp_item = val_unit_pair.popitem()
        assert conv_item[0] == exp_item[0]
        assert conv_item[1]['unit'] == exp_item[1]['unit']
        assert_array_equal(conv_item[1]['val'], exp_item[1]['val'])
        
    def test_nested_dictionaries_are_converted_correctly(self):
        test = dict(a=dict(a1=1 * ureg.ms,
                           a2=[1, 2, 3] * ureg.Hz),
                    b=dict(b1=1 * ureg.Hz,
                           b2=2 * ureg.ms),
                    c=3 * ureg.m)
        converted = io.quantities_to_val_unit(test)
        check_dict_contains_no_quantity(converted)
        assert isinstance(converted['a']['a1'], dict)
        assert isinstance(converted['a']['a2'], dict)
        assert isinstance(converted['b']['b1'], dict)
        assert isinstance(converted['b']['b2'], dict)
        assert isinstance(converted['c'], dict)
        
        
class Test_convert_arrays_in_dict_to_list:
    
    def test_converts_simple_dict(self):
        tdict = {'a': np.array([1, 2, 3]),
                 'b': [3, 4, 5],
                 'c': 'spam',
                 'd': 1}
        converted = io.convert_arrays_in_dict_to_lists(tdict)
        tdict.update({'a': [1, 2, 3]})
        check_quantity_dicts_are_equal(converted, tdict)
        
    def test_converts_nested_dict(self):
        tdict = {'a': {'val': np.array([1, 2, 3]), 'unit': 'ms'},
                 'b': [3, 4, 5],
                 'c': 'spam',
                 'd': 1}
        converted = io.convert_arrays_in_dict_to_lists(tdict)
        tdict.update({'a': {'val': [1, 2, 3], 'unit': 'ms'}})
        check_quantity_dicts_are_equal(converted, tdict)
        
        
class Test_save_quantity_dict_to_yaml:
    
    def test_quantities_to_val_unit_called(self, mocker, tmpdir,
                                           network_dict_quantity):
        file = 'test.yaml'
        mock = mocker.patch('nnmt.input_output.'
                            'quantities_to_val_unit',
                            return_value=network_dict_quantity)
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            io.save_quantity_dict_to_yaml(file, network_dict_quantity)
        mock.assert_called_once()

    def test_quantity_dict_saved_correctly(self, tmpdir,
                                           network_dict_quantity):
        file = 'test.yaml'
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            io.save_quantity_dict_to_yaml(file, network_dict_quantity)
            with open(file, 'r') as stream:
                loaded = yaml.safe_load(stream)
            check_dict_contains_no_quantity(loaded)
            # check that dicts are not empty
            for sdict in loaded:
                assert bool(sdict)
            
            
class Test_load_val_unit_dict_from_yaml:

    def test_val_unit_to_quantities_called(self, mocker):
        mock = mocker.patch('nnmt.input_output.'
                            'val_unit_to_quantities')
        io.load_val_unit_dict_from_yaml(f'{path_to_fixtures}test.yaml')
        mock.assert_called_once()

    def test_yaml_loaded_correctly(self, param_test_dict):
        params = io.load_val_unit_dict_from_yaml(
            f'{path_to_fixtures}test.yaml')
        check_quantity_dicts_are_equal(params, param_test_dict)


class Test_save_network:

    def test_h5_is_created(self, tmpdir, network):
        tmp_test = tmpdir.mkdir('tmp_test')
        file = 'test.h5'
        with tmp_test.as_cwd():
            io.save_network(file, network)
        check_file_in_tmpdir(file, tmp_test)

    def test_save_overwriting_existing_file_raises_error(self, tmpdir,
                                                         network):
        file = 'test.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            with pytest.raises(IOError):
                io.save_network(file, network)
                io.save_network(file, network)
            
    def test_save_creates_correct_output(self, tmpdir, empty_network):
        file = 'test.h5'
        keys = ['results', 'results_hash_dict', 'network_params',
                'analysis_params']
        
        def _test_func(x):
            return x
    
        def test_function(network):
            return nnmt.utils._cache(network, _test_func, {'x': 1}, 'test')
    
        empty_network.network_params['a'] = 1
        empty_network.analysis_params['a'] = 1
        test_function(empty_network)
        
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            io.save_network(file, empty_network, overwrite=True)
            output = h5.load(file)
            for key in keys:
                assert key in output.keys()
            # check that dicts are not empty
            for sub_dict in output.values():
                assert bool(sub_dict)
            # check that all quantities have been converted
            check_dict_contains_no_quantity(output)
            
                
class Test_load_network:
    
    def test_warning_is_raised_if_file_doesnt_exist(self, tmpdir):
        file = 'test.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            with pytest.warns(UserWarning):
                io.load_network(file)
                
    def test_returns_empty_dicts_if_no_file_present(self, tmpdir):
        file = 'test.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                outputs = io.load_network(file)
            for output in outputs:
                assert not bool(output)
            
    def test_input_is_converted_to_quantities(self, tmpdir,
                                              network_dict_val_unit):
        file = 'test.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            h5.save(file, network_dict_val_unit)
            outputs = io.load_network(file)
            # check that all val unit dicts have been converted to quantities
            for output in outputs:
                check_dict_contains_no_val_unit_dict(output)
        
    def test_loaded_dictionaries_are_not_empty(self, tmpdir,
                                               network_dict_val_unit):
        file = 'test.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            h5.save(file, network_dict_val_unit)
            outputs = io.load_network(file)
            # check that no loaded dictionary is empty
            for sub_dict in outputs:
                assert bool(sub_dict)
            
    def test_returns_dictionaries_in_correct_order(self, tmpdir,
                                                   network_dict_val_unit):
        file = 'test.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            h5.save(file, network_dict_val_unit)
            outputs = io.load_network(file)
            assert 'tau_m' in outputs[0].keys()
            assert 'omegas' in outputs[1].keys()
            assert 'test' in outputs[2].keys()
            rhd = [dict for dict in outputs[3].values()]
            assert 'test' in rhd[0].keys()
            assert 'analysis_params' in rhd[0].keys()
            

class Test_save_quantity_dict_to_h5:
    
    def test_h5_is_created(self, tmpdir, network_dict_quantity):
        file = 'test.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            io.save_quantity_dict_to_h5(file, network_dict_quantity)
        check_file_in_tmpdir(file, tmp_test)

    def test_save_overwriting_existing_file_raises_error(
            self, tmpdir, network_dict_quantity):
        file = 'test.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            with pytest.raises(IOError):
                io.save_quantity_dict_to_h5(file, network_dict_quantity)
                io.save_quantity_dict_to_h5(file, network_dict_quantity)
                
    def test_save_creates_correct_output(self, tmpdir, network_dict_quantity):
        file = 'test.h5'
        keys = ['results', 'results_hash_dict', 'network_params',
                'analysis_params']
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            io.save_quantity_dict_to_h5(file, network_dict_quantity)
            output = h5.load(file)
            for key in keys:
                assert key in output.keys()
            # check that dicts are not empty
            for sub_dict in output.values():
                assert bool(sub_dict)
            # check that all quantities have been converted
            check_dict_contains_no_quantity(output)


class Test_load_val_unit_dict_from_h5:

    def test_raise_exception_if_filename_not_existing(self, tmpdir):
        tmp_test = tmpdir.mkdir('tmp_test')
        file = 'test.h5'
        with tmp_test.as_cwd():
            with pytest.raises(IOError):
                io.load_val_unit_dict_from_h5(file)

    def test_input_is_converted_to_quantities(self, tmpdir,
                                              network_dict_val_unit):
        file = 'test.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            h5.save(file, network_dict_val_unit)
            output = io.load_val_unit_dict_from_h5(file)
            # check that all val unit dicts have been converted to quantities
            check_dict_contains_no_val_unit_dict(output)
            
    def test_loaded_dictionaries_are_not_empty(self, tmpdir,
                                               network_dict_val_unit):
        file = 'test.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            h5.save(file, network_dict_val_unit)
            output = io.load_val_unit_dict_from_h5(file)
            # check that no loaded dictionary is empty
            assert bool(output)


class Test_create_hash:

    def test_correct_hash(self):
        params = dict(a=1, b=2)
        hash = io.create_hash(params, ['a', 'b'])
        assert hash == 'c20ad4d76fe97759aa27a0c99bff6710'

    def test_hash_only_reflects_given_keys(self):
        params = dict(a=1, b=2, c=3)
        hash = io.create_hash(params, ['a', 'b'])
        assert hash == 'c20ad4d76fe97759aa27a0c99bff6710'
