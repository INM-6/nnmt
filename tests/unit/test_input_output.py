# -*- coding: utf-8 -*-
"""
Unit tests for the input output module.
"""

import pytest
import numpy as np

import lif_meanfield_tools as lmt
import lif_meanfield_tools.input_output as io

from .checks import check_file_in_tmpdir, check_quantity_dicts_are_equal

ureg = lmt.ureg
        
val_unit_pairs = [
    dict(numerical=1),
    dict(quantity={'val': 1, 'unit': 'hertz'}),
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
                assert conv_item == exp_item
            except ValueError:
                assert conv_item[0] == exp_item[0]
                np.testing.assert_array_equal(conv_item[1], exp_item[1])
            
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
            assert conv_item[0] == exp_item[0]
            print(conv_item)
            try:
                assert conv_item[1]['unit'] == exp_item[1]['unit']
                try:
                    assert conv_item[1]['val'] == exp_item[1]['val']
                except ValueError:
                    np.testing.assert_array_equal(conv_item[1]['val'],
                                                  exp_item[1]['val'])
            except (IndexError, TypeError):
                try:
                    assert conv_item[1] == conv_item[1]
                except ValueError:
                    np.testing.assert_array_equal(conv_item[1], exp_item[1])
                    
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
        np.testing.assert_array_equal(conv_item[1]['val'], exp_item[1]['val'])
        
    @pytest.mark.xfail
    def test_list_of_quantities_with_several_units_raises_exception(self):
        quantity_dict = dict(list_of_quantities=[1 * ureg.Hz,
                                                 2 * ureg.s,
                                                 3 * ureg.m])
        with pytest.raises(ValueError):
            io.quantities_to_val_unit(quantity_dict)


class Test_load_params:
    
    def test_val_unit_to_quantities_called(self, mocker):
        mock = mocker.patch('lif_meanfield_tools.input_output.'
                            'val_unit_to_quantities')
        io.load_params('tests/fixtures/test.yaml')
        mock.assert_called_once()
        
    def test_yaml_loaded_correctly(self, param_test_dict):
        params = io.load_params('tests/fixtures/test.yaml')
        check_quantity_dicts_are_equal(params, param_test_dict)
                

class Test_save:
    
    def test_h5_is_created(self, tmpdir, param_test_dict):
        tmp_test = tmpdir.mkdir('tmp_test')
        output_key = 'params'
        file_name = 'test.h5'
        with tmp_test.as_cwd():
            io.save(output_key, param_test_dict, file_name)
        check_file_in_tmpdir(file_name, tmp_test)
        
    @pytest.mark.xfail
    def test_save_overwriting_existing_file_raises_error(self, tmpdir,
                                                         param_test_dict):
        file_name = 'test.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        output_key = 'params'
        file_name = 'test.h5'
        with tmp_test.as_cwd():
            with pytest.raises(IOError):
                io.save(output_key, param_test_dict, file_name)
                io.save(output_key, param_test_dict, file_name)


class Test_create_hash:
    
    def test_correct_hash(self):
        params = dict(a=1, b=2)
        hash = io.create_hash(params, ['a', 'b'])
        assert hash == 'c20ad4d76fe97759aa27a0c99bff6710'
        
    def test_hash_only_reflects_given_keys(self):
        params = dict(a=1, b=2, c=3)
        hash = io.create_hash(params, ['a', 'b'])
        assert hash == 'c20ad4d76fe97759aa27a0c99bff6710'


class Test_load_h5:
    
    @pytest.mark.xfail
    def test_raise_exception_if_filename_not_existing(self, tmpdir):
        tmp_test = tmpdir.mkdir('tmp_test')
        filename = 'test.h5'
        with tmp_test.as_cwd():
            with pytest.raises(IOError):
                io.load_h5(filename)
    
    def test_data_saved_and_loaded_correctly(self, tmpdir, param_test_dict):
        tmp_test = tmpdir.mkdir('tmp_test')
        filename = 'test.h5'
        with tmp_test.as_cwd():
            io.save('params', param_test_dict, filename)
            params_h5 = io.load_h5(filename)
            
        params = params_h5['params']
        check_quantity_dicts_are_equal(params, param_test_dict)

#
# class save_and_load_TestCase(unittest.TestCase):
#
#     def setUp(self):
#         self.test_dir = './lif_meanfield_tools/tests/unit/'
#         self.file_name = 'temporary_test_file.h5'
#         self.file_name_with_dir = self.test_dir + self.file_name
#
#         self.output_key = 'test_results'
#
#         # integer
#         self.quantity_1 = 1 * ureg.Hz
#         # array
#         self.quantity_2 = np.array([2, 3, 4]) * ureg.mV
#         # list
#         self.quantity_3 = [5, 6, 7] * ureg.s
#         # list of quantities
#         self.list_of_quantites = [5 * ureg.s, 6.7 * ureg.m]
#         # list of strings
#         self.list_of_strings = ['list', 'of', 'strings']
#         # no quantity
#         self.no_quantity = 8
#
#         # build dictionary
#         self.output = {'quantity_1': self.quantity_1,
#                        'quantity_2': self.quantity_2,
#                        'quantity_3': self.quantity_3,
#                        'list_of_quantites': self.list_of_quantites,
#                        'list_of_strings': self.list_of_strings,
#                        'no_quantity': self.no_quantity}
#
#     def test_save_and_load(self):
#         lmt.input_output.save(self.output_key,
#                               self.output,
#                               self.file_name_with_dir)
#
#         # check that file with correct file_name exists
#         self.assertTrue(self.file_name in os.listdir(self.test_dir))
#
#         loaded_data = lmt.input_output.load_h5(self.file_name_with_dir)
#
#         self.assertListEqual(sorted(list(loaded_data[self.output_key].keys())),
#                              sorted(list(self.output.keys())))
#
#         for val in loaded_data[self.output_key].values():
#             self.assertTrue(val is not None)
#             # checking the actual elements and values is covered by
#             # val_unit_to_quantities
#
#     def tearDown(self):
#         os.remove(self.test_dir + self.file_name)


if __name__ == '__main__':
    unittest.main()
