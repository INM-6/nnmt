# -*- coding: utf-8 -*-
"""
Unit tests for the input output module.
"""

import unittest
import os

import pytest

import numpy as np
from numpy.testing import assert_array_equal

import lif_meanfield_tools as lmt
import lif_meanfield_tools.input_output as io

ureg = lmt.ureg
        
val_unit_pairs = [
    dict(numerical=1),
    dict(quantity={'val': 1, 'unit': 'Hz'}),
    dict(string='test'),
    dict(list_of_strings=['spam', 'ham']),
    dict(array=np.array([1, 2, 3])),
    dict(quantity_array={'val': np.array([1, 2, 3]), 'unit': 's'}),
    dict(list_of_quantities={'val': [1, 2, 3], 'unit': 's'}),
    dict(two_d_array=np.arange(9).reshape(3, 3)),
    dict(two_d_quantity_array={'val': np.arange(9).reshape(3, 3),
                               'unit': 's'}),
    dict(two_d_list_of_quantites={'val': [[i for i in range(3)]
                                          for j in range(3)],
                                  'unit': 's'}),
    dict(numerical=1,
         quantity={'val': 1, 'unit': 'Hz'},
         string='test',
         list_of_strings=['spam', 'ham'],
         array=np.array([1, 2, 3]),
         quantity_array={'val': np.array([1, 2, 3]), 'unit': 's'},
         list_of_quantities={'val': [1, 2, 3], 'unit': 's'},
         two_d_array=np.arange(9).reshape(3, 3),
         two_d_quantity_array={'val': np.arange(9).reshape(3, 3),
                               'unit': 's'},
         two_d_list_of_quantites={'val': [[i for i in range(3)]
                                          for j in range(3)],
                                  'unit': 's'},
         ),
    ]

quantity_dicts = [
    dict(numerical=1),
    dict(quantity=1 * ureg.Hz),
    dict(string='test'),
    dict(list_of_strings=['spam', 'ham']),
    dict(array=np.array([1, 2, 3])),
    dict(quantity_array=np.array([1, 2, 3]) * ureg.s),
    dict(list_of_quantities=np.array([1, 2, 3]) * ureg.s),
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
         list_of_quantities=np.array([1, 2, 3]) * ureg.s,
         two_d_array=np.arange(9).reshape(3, 3),
         two_d_quantity_array=np.arange(9).reshape(3, 3) * ureg.s,
         two_d_list_of_quantites=[[i for i in range(3)]
                                  for j in range(3)] * ureg.s),
    ]

ids = [list(val_unit_pair.keys())[0] for val_unit_pair in val_unit_pairs[:-1]]
ids.append('mixed')


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


class quantities_to_val_unit_TestCase(unittest.TestCase):
    
    def setUp(self):
        # integer
        self.quantity_1 = 1 * ureg.Hz
        # array
        self.quantity_2 = np.array([2, 3, 4]) * ureg.mV
        # list
        self.quantity_3 = [5, 6, 7] * ureg.s
        # list of quantities
        self.list_of_quantites = [5 * ureg.s, 6.7 * ureg.m]
        # list of strings
        self.list_of_strings = ['list', 'of', 'strings']
        # no quantity
        self.no_quantity = 8

        # build dictionary
        self.test_quantity_dict = {'quantity_1': self.quantity_1,
                                   'quantity_2': self.quantity_2,
                                   'quantity_3': self.quantity_3,
                                   'list_of_quantites': self.list_of_quantites,
                                   'list_of_strings': self.list_of_strings,
                                   'no_quantity': self.no_quantity}

    def test_quantity_to_val_unit(self):
        val_unit_dict = lmt.input_output.quantities_to_val_unit(
            self.test_quantity_dict)

        # integer
        self.assertEqual(val_unit_dict['quantity_1']['unit'],
                         self.quantity_1.units)
        assert_array_equal(val_unit_dict['quantity_1']['val'],
                           self.quantity_1.magnitude)
        # array
        self.assertEqual(val_unit_dict['quantity_2']['unit'],
                         self.quantity_2.units)
        assert_array_equal(val_unit_dict['quantity_2']['val'],
                           self.quantity_2.magnitude)
        # list
        self.assertNotIsInstance(self.test_quantity_dict['quantity_3'], list)
        self.assertIsInstance(self.test_quantity_dict['quantity_3'],
                              ureg.Quantity)
        self.assertEqual(val_unit_dict['quantity_3']['unit'],
                         self.quantity_3.units)
        assert_array_equal(val_unit_dict['quantity_3']['val'],
                           self.quantity_3.magnitude)

        # list of quantities
        self.assertIsInstance(self.list_of_quantites, list)
        self.assertFalse(any(isinstance(part, str) for part
                             in self.list_of_quantites))
        # in this case the magnitudes are stacked
        assert_array_equal(val_unit_dict['list_of_quantites']['val'],
                           np.stack(val.magnitude for val
                                    in self.list_of_quantites))
        # and only the first appearing quantity-unit is taken
        self.assertEqual(val_unit_dict['list_of_quantites']['unit'],
                         self.list_of_quantites[0].units)

        # list of strings
        self.assertIsInstance(self.list_of_strings, list)
        self.assertTrue(any(isinstance(part, str) for part
                            in self.list_of_strings))
        assert_array_equal(val_unit_dict['list_of_strings'],
                           self.list_of_strings)

        # no quantity
        self.assertEqual(val_unit_dict['no_quantity'], self.no_quantity)


class save_and_load_TestCase(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = './lif_meanfield_tools/tests/unit/'
        self.file_name = 'temporary_test_file.h5'
        self.file_name_with_dir = self.test_dir + self.file_name

        self.output_key = 'test_results'

        # integer
        self.quantity_1 = 1 * ureg.Hz
        # array
        self.quantity_2 = np.array([2, 3, 4]) * ureg.mV
        # list
        self.quantity_3 = [5, 6, 7] * ureg.s
        # list of quantities
        self.list_of_quantites = [5 * ureg.s, 6.7 * ureg.m]
        # list of strings
        self.list_of_strings = ['list', 'of', 'strings']
        # no quantity
        self.no_quantity = 8

        # build dictionary
        self.output = {'quantity_1': self.quantity_1,
                       'quantity_2': self.quantity_2,
                       'quantity_3': self.quantity_3,
                       'list_of_quantites': self.list_of_quantites,
                       'list_of_strings': self.list_of_strings,
                       'no_quantity': self.no_quantity}

    def test_save_and_load(self):
        lmt.input_output.save(self.output_key,
                              self.output,
                              self.file_name_with_dir)

        # check that file with correct file_name exists
        self.assertTrue(self.file_name in os.listdir(self.test_dir))

        loaded_data = lmt.input_output.load_h5(self.file_name_with_dir)

        self.assertListEqual(sorted(list(loaded_data[self.output_key].keys())),
                             sorted(list(self.output.keys())))

        for val in loaded_data[self.output_key].values():
            self.assertTrue(val is not None)
            # checking the actual elements and values is covered by
            # val_unit_to_quantities

    def tearDown(self):
        os.remove(self.test_dir + self.file_name)


if __name__ == '__main__':
    unittest.main()
