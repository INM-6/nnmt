import pytest
import numpy as np
from numpy.testing import (
    assert_array_equal)

from ..checks import (
    assert_units_equal,
    check_quantity_dicts_are_equal,
    )

import lif_meanfield_tools as lmt
ureg = lmt.ureg


def make_test_method(output):
    @lmt.utils._check_and_store(['test'])
    def test_method(self):
        return output
    return test_method


def make_cache_test_func(output):
    def cache_test_func(network):
        params = dict(output=output)
        return lmt.utils._cache(_test_func, params, 'test', network)
    return cache_test_func
        
        
def _test_func(output):
    return output


def make_test_method_with_key(output, key):
    @lmt.utils._check_and_store(['test'], key)
    def test_method(self, key):
        return output
    return test_method


result_types = dict(numerical=1,
                    quantity=1 * ureg.s,
                    array=np.array([1, 2, 3]),
                    quantity_array=np.array([1, 2, 3]) * ureg.s,
                    two_d_array=np.arange(9).reshape(3, 3),
                    two_d_quantity_array=np.arange(9).reshape(3, 3) * ureg.s,
                    )
analysis_key_types = dict(numerical=1,
                          quantity=1 * ureg.s,
                          string='test_string',
                          array=np.array([1, 2, 3]),
                          quantity_array=np.array([1, 2, 3]) * ureg.s,
                          )
result_ids = sorted(result_types.keys())
key_names = sorted(analysis_key_types.keys())
test_methods = [make_test_method(result_types[key])
                for key in result_ids]
keys = [analysis_key_types[key] for key in key_names]
results = [result_types[key] for key in result_ids]

cache_test_funcs = [make_cache_test_func(result_types[key])
                    for key in result_ids]


class Test_cache:
    
    @pytest.mark.parametrize('test_func, result',
                             zip(cache_test_funcs, results),
                             ids=result_ids)
    def test_save_results_in_network_dicts(self, network, test_func, result):
        test_func(network)
        try:
            assert network.results['test'] == result
        except ValueError:
            assert_array_equal(network.results['test'], result)
            assert_units_equal(network.results['test'], result)
        
        rhd_entry = network.results_hash_dict.popitem()[1]
        
        assert 'test' in rhd_entry
        try:
            assert rhd_entry['test'] == result
        except ValueError:
            assert_array_equal(rhd_entry['test'], result)
            assert_units_equal(rhd_entry['test'], result)
        
        assert 'params' in rhd_entry
        check_quantity_dicts_are_equal(rhd_entry['params'],
                                       dict(output=result))
            
    @pytest.mark.parametrize('test_func, result',
                             zip(cache_test_funcs, results),
                             ids=result_ids)
    def test_returns_existing_result(self, network, test_func,
                                     result):
        test_func(network)
        try:
            assert test_func(network) == result
        except ValueError:
            assert_array_equal(test_func(network), result)
            assert_units_equal(test_func(network), result)

    def test_result_not_calculated_twice(self, mocker, network):
        mock = mocker.Mock(__name__='mocker', return_value=1)
        
        def test_function(network):
            return lmt.utils._cache(mock, dict(a=1), 'test', network)
        
        test_function(network)
        test_function(network)
        mock.assert_called_once()
        
    def test_result_calculated_twice_for_differing_params(self, mocker,
                                                          empty_network):
        network = empty_network
        mock = mocker.Mock(__name__='mocker')
        
        def test_function(network, a):
            params = dict(a=a)
            return lmt.utils._cache(mock, params, 'test', network)
        
        test_function(network, 1)
        test_function(network, 2)
        assert mock.call_count == 2
        
    def test_result_calculated_twice_for_differing_result_keys(self, mocker,
                                                               empty_network):
        network = empty_network
        mock = mocker.Mock(__name__='mocker')
        
        def test_function(network, key):
            return lmt.utils._cache(mock, dict(a=1), key, network)
        
        test_function(network, 'test1')
        test_function(network, 'test2')
        assert mock.call_count == 2
