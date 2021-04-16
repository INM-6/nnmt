# -*- coding: utf-8 -*-
"""
Unit and integration tests for the h5py_wrapper module

"""

import unittest
import wrapper
import numpy
from numpy.testing import assert_array_equal

fn = 'data.h5'

# define data
i0 = 6
f0 = 3.14159
s0 = 'this is a test'

a0i = [1, 2, 3, 4, 5]
a0s = ['a', 'b', 'c']
m0 = [[6, 7, 8], [9, 10, 11]]
# an0 = [[12,13],[14,15,16]] #NESTED ARRAY FAILS DUE TO UNKOWN OBJECT TYPE

d0 = {'i': i0, 'f': f0, 's': s0}
dn0 = {'d1': d0, 'd2': d0}

# define containers
simpledata_str = ['i', 'f', 's']
simpledata_val = [i0, f0, s0]

arraydata_str = ['ai', 'as', 'm']
arraydata_val = [a0i, a0s, m0]

dictdata_str = ['d']
dictdata_val = [d0]


class WrapperTest(unittest.TestCase):
    def construct_simpledata(self):
        res = {}
        for key, val in zip(simpledata_str, simpledata_val):
            res[key] = val
        return res
    
    def test_write_and_load_with_label(self):
        res = self.construct_simpledata()
        wrapper.add_to_h5(fn, res, write_mode='w', dict_label='test_label')
        for key, val in zip(simpledata_str, simpledata_val):
            assert(wrapper.load_h5(fn, 'test_label/' + key) == val)

    def test_store_and_load_dataset_directly(self):
        res = self.construct_simpledata()
        wrapper.add_to_h5(fn, res, write_mode='w')
        for key, val in zip(simpledata_str, simpledata_val):
            assert(wrapper.load_h5(fn, '/' + key) == val)

    def test_old_store_and_load_simpledata(self):
        res = self.construct_simpledata()
        wrapper.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = wrapper.load_h5(fn)
        for key, val in zip(simpledata_str, simpledata_val):
            assert(res[key] == val)

    def test_store_and_load_simpledata(self):
        res = self.construct_simpledata()
        wrapper.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = wrapper.load_h5(fn)
        for key, val in zip(simpledata_str, simpledata_val):
            assert(res[key] == val)

    def test_store_and_load_arraydata(self):
        res = {}
        for key, val in zip(arraydata_str, arraydata_val):
            res[key] = val
        wrapper.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = wrapper.load_h5(fn)
        for key, val in zip(arraydata_str, arraydata_val):
            assert_array_equal(res[key], val)

    def test_store_and_load_dictdata(self):
        res = {}
        for key, val in zip(dictdata_str, dictdata_val):
            res[key] = val
        wrapper.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = wrapper.load_h5(fn)
        for dkey, dval in zip(dictdata_str, dictdata_val):
            for key, val in dval.items():
                assert(res[dkey][key] == val)

    def test_check_for_node(self):
        res = {'a': 1, 'test1': {'b': 2}, 'test2': {'test3': {'c': 3}}}
        wrapper.add_to_h5(fn, res, write_mode='w')
        assert(wrapper.node_exists(fn, '/a'))
        assert(wrapper.node_exists(fn, '/nota') is False)
        assert(wrapper.node_exists(fn, '/test1/b'))
        assert(wrapper.node_exists(fn, '/test1/notb') is False)
        assert(wrapper.node_exists(fn, '/test2/test3/c'))
        assert(wrapper.node_exists(fn, '/test2/test3/notc') is False)

    def test_overwrite_dataset(self):
        res = {'a': 5}
        wrapper.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = {'a': 6}
        wrapper.add_to_h5(
            fn, res, write_mode='a', overwrite_dataset=False)
        res.clear()
        res = wrapper.load_h5(fn)
        assert(res['a'] == 5)  # dataset should still contain old value
        res.clear()
        res = {'a': 6}
        wrapper.add_to_h5(
            fn, res, write_mode='a', overwrite_dataset=True)
        res.clear()
        res = wrapper.load_h5(fn)
        assert(res['a'] == 6)  # dataset should contain new value

    def test_write_empty_array(self):
        res = {'a': [], 'b': numpy.array([])}
        wrapper.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = wrapper.load_h5(fn)
        assert_array_equal(res['a'], [])
        assert_array_equal(res['b'], [])

    def test_write_nested_empty_array(self):
        res = {'a': [[], []], 'b': numpy.array([[], []])}
        wrapper.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = wrapper.load_h5(fn)
        assert_array_equal(res['a'], [[], []])
        assert(numpy.shape(res['a']) == (2, 0))
        assert_array_equal(res['b'], [[], []])
        assert(numpy.shape(res['b']) == (2, 0))

    def test_read_empty_array_via_path(self):
        res = {'a': numpy.array([[], []])}
        wrapper.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = wrapper.load_h5(fn, path='a')
        assert_array_equal(res, [[], []])
        assert(numpy.shape(res) == (2, 0))

    def test_handle_nonexisting_path(self):
        res = {}
        stest = 'this is a test'
        wrapper.add_to_h5(fn, res, write_mode='w')
        try:
            res = wrapper.load_h5(fn, path='test/')
            raise Exception()  # should not get until here
        except KeyError:
            res['test'] = stest
            wrapper.add_to_h5(fn, res)
            res.clear()
            res = wrapper.load_h5(fn, path='test/')
            assert(res == stest)

    def test_store_none(self):
        res = {'a1': None}
        wrapper.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = wrapper.load_h5(fn)
        assert(res['a1'] is None)

    def test_handle_nonexisting_file(self):
        try:
            wrapper.load_h5('asdasd.h5')
            raise Exception()  # should not get until here
        except IOError:
            pass

    def test_store_and_load_custom_array(self):
        a = [[1, 2, 3, 4], [6, 7]]
        wrapper.add_to_h5(fn, {'a': a}, overwrite_dataset=True)
        # loading the whole data
        res = wrapper.load_h5(fn)
        for i in xrange(len(a)):
            assert(numpy.sum(a[i] - res['a'][i]) < 1e-12)
        # loading path directly
        res = wrapper.load_h5(fn, path='a/')
        for i in xrange(len(a)):
            assert(numpy.sum(a[i] - res[i]) < 1e-12)

    def test_store_and_load_quantities_array(self):
        import quantities as pq
        data = {'times': numpy.array(
            [1, 2, 3]) * pq.ms, 'positions': numpy.array([1, 2, 3]) * pq.cm}
        wrapper.add_to_h5(fn, data, overwrite_dataset=True)
        # loading the whole data
        res = wrapper.load_h5(fn)
        assert(res['times'].dimensionality == data['times'].dimensionality)

    def test_store_and_load_with_compression(self):
        data = {'a': 1, 'test1': {'b': 2}, 'test2': {
            'test3': {'c': numpy.array([1, 2, 3])}}}
        wrapper.add_to_h5(fn, data, write_mode='w', compression='gzip')
        wrapper.load_h5(fn)

    def test_store_and_test_key_types(self):
        data = {'a': 1, (1, 2): 2., 4.: 3.}
        wrapper.add_to_h5(fn, data, write_mode='w', compression='gzip')
        res = wrapper.load_h5(fn)

        keys = ['a', (1, 2), 4.]
        for k in keys:
            assert(k in res.keys())

    def test_load_lazy_simple(self):
        res = self.construct_simpledata()
        wrapper.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = wrapper.load_h5(fn, lazy=True)
        for key, obj in res.items():
            assert(obj is None)

    def test_load_lazy_nested(self):
        res = {'a': 1, 'test1': {'b': 2}, 'test2': {
            'test3': {'c': numpy.array([1, 2, 3])}}}
        wrapper.add_to_h5(fn, res, write_mode='w')
        res.clear()
        res = wrapper.load_h5(fn, lazy=True)
        assert(res['a'] is None)
        assert(res['test1']['b'] is None)
        assert(res['test2']['test3']['c'] is None)
