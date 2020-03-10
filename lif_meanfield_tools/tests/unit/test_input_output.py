# -*- coding: utf-8 -*-
"""
Unit tests for the input output module.
"""

import unittest
import os

import numpy as np
from numpy.testing import assert_array_equal

import lif_meanfield_tools as lmt
from ... import ureg
import h5py_wrapper.wrapper as h5

class val_unit_to_quantities_TestCase(unittest.TestCase):
    def setUp(self):
        # integer
        self.val_unit_1 = {'val': 1,
                           'unit': 'Hz'}
        # array
        self.val_unit_2 = {'val': np.array([2,3,4]),
                           'unit': 'mV'}
        # list
        self.val_unit_3 = {'val': [5,6,7],
                           'unit': 's'}
        # no unit given
        self.val_unit_4 = {'val': 8}

        # build dictionary
        self.test_val_unit_dict = {'val_unit_1': self.val_unit_1,
                                           'val_unit_2': self.val_unit_2,
                                           'val_unit_3': self.val_unit_3,
                                           'val_unit_4': self.val_unit_4}

    def test_val_unit_to_quantities(self):
        quantity_dict = lmt.input_output.val_unit_to_quantities(self.test_val_unit_dict)

        # integer
        self.assertIsInstance(quantity_dict['val_unit_1'].magnitude, int)
        self.assertEqual(quantity_dict['val_unit_1'].magnitude, self.val_unit_1['val'])
        self.assertEqual(quantity_dict['val_unit_1'].units, ureg.parse_expression(self.val_unit_1['unit']))

        # array
        self.assertIsInstance(quantity_dict['val_unit_2'].magnitude, np.ndarray)
        assert_array_equal(quantity_dict['val_unit_2'].magnitude, self.val_unit_2['val'])
        self.assertEqual(quantity_dict['val_unit_2'].units, ureg.parse_expression(self.val_unit_2['unit']))

        # list
        self.assertIsInstance(quantity_dict['val_unit_3'].magnitude, np.ndarray)
        assert_array_equal(list(quantity_dict['val_unit_3'].magnitude), self.val_unit_3['val'])
        self.assertEqual(quantity_dict['val_unit_3'].units, ureg.parse_expression(self.val_unit_3['unit']))

        #  no unit given
        self.assertIsInstance(quantity_dict['val_unit_4'], dict)
        self.assertEqual(quantity_dict['val_unit_4']['val'], self.val_unit_4['val'])

class quantities_to_val_unit_TestCase(unittest.TestCase):
    def setUp(self):
        # integer
        self.quantity_1 = 1*ureg.Hz
        # array
        self.quantity_2 = np.array([2,3,4])*ureg.mV
        # list
        self.quantity_3 = [5,6,7]*ureg.s
        # list of quantities
        self.list_of_quantites = [5*ureg.s, 6.7*ureg.m]
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
        val_unit_dict = lmt.input_output.quantities_to_val_unit(self.test_quantity_dict)

        # integer
        self.assertEqual(val_unit_dict['quantity_1']['unit'], self.quantity_1.units)
        assert_array_equal(val_unit_dict['quantity_1']['val'], self.quantity_1.magnitude)
        # array
        self.assertEqual(val_unit_dict['quantity_2']['unit'], self.quantity_2.units)
        assert_array_equal(val_unit_dict['quantity_2']['val'], self.quantity_2.magnitude)
        # list
        self.assertNotIsInstance(self.test_quantity_dict['quantity_3'], list)
        self.assertIsInstance(self.test_quantity_dict['quantity_3'], ureg.Quantity)
        self.assertEqual(val_unit_dict['quantity_3']['unit'], self.quantity_3.units)
        assert_array_equal(val_unit_dict['quantity_3']['val'], self.quantity_3.magnitude)

        # list of quantities
        self.assertIsInstance(self.list_of_quantites, list)
        self.assertFalse(any(isinstance(part, str) for part in self.list_of_quantites))
        # in this case the magnitudes are stacked
        assert_array_equal(val_unit_dict['list_of_quantites']['val'],
                           np.stack(val.magnitude for val in self.list_of_quantites))
        # and only the first appearing quantity-unit is taken
        self.assertEqual(val_unit_dict['list_of_quantites']['unit'], self.list_of_quantites[0].units)

        # list of strings
        self.assertIsInstance(self.list_of_strings, list)
        self.assertTrue(any(isinstance(part, str) for part in self.list_of_strings))
        assert_array_equal(val_unit_dict['list_of_strings'], self.list_of_strings)


        # no quantity
        self.assertEqual(val_unit_dict['no_quantity'], self.no_quantity)


class save_and_load_TestCase(unittest.TestCase):
    def setUp(self):
        self.test_dir = './lif_meanfield_tools/tests/unit/'
        self.file_name = 'temporary_test_file.h5'
        self.file_name_with_dir = self.test_dir + self.file_name

        self.output_key = 'test_results'

        # integer
        self.quantity_1 = 1*ureg.Hz
        # array
        self.quantity_2 = np.array([2,3,4])*ureg.mV
        # list
        self.quantity_3 = [5,6,7]*ureg.s
        # list of quantities
        self.list_of_quantites = [5*ureg.s, 6.7*ureg.m]
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
            # checking the actual elements and values is covered by val_unit_to_quantities

    def tearDown(self):
        os.remove(self.test_dir + self.file_name)

if __name__ == '__main__':
    unittest.main()
