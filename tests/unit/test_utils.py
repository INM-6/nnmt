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


class Test_check_and_store_decorator:

    @pytest.mark.parametrize('test_method, result', zip(test_methods, results),
                             ids=result_ids)
    def test_save_results(self, network, test_method, result):
        test_method(network)
        try:
            assert network.results['test'] == result
        except ValueError:
            assert_array_equal(network.results['test'], result)
            assert_units_equal(network.results['test'], result)

    @pytest.mark.parametrize('test_method, result', zip(test_methods, results),
                             ids=result_ids)
    def test_returns_existing_result(self, network, test_method,
                                     result):
        test_method(network)
        try:
            assert test_method(network) == result
        except ValueError:
            assert_array_equal(test_method(network), result)
            assert_units_equal(test_method(network), result)

    def test_result_not_calculated_twice(self, mocker, network):
        mock = mocker.Mock()
        
        @lmt.utils._check_and_store(['test'])
        def test_function(network):
            return mock()
        
        test_function(network)
        test_function(network)
        mock.assert_called_once()

    @pytest.mark.parametrize('key', keys, ids=key_names)
    @pytest.mark.parametrize('result', results, ids=result_ids)
    def test_saves_new_analysis_key_with_param_and_results(self,
                                                           network,
                                                           key,
                                                           result):
        test_method = make_test_method_with_key(result, ['test_key'])
        test_method(network, key)
        try:
            assert network.results['test'] == result
        except ValueError:
            assert_array_equal(network.results['test'], result)
            assert_units_equal(network.results['test'], result)

    @pytest.mark.parametrize('key', keys, ids=key_names)
    @pytest.mark.parametrize('result', results, ids=result_ids)
    def test_returns_existing_analysis_key_with_param_and_results(self,
                                                                  network,
                                                                  key,
                                                                  result):
        test_method = make_test_method_with_key(result, ['test_key'])
        test_method(network, key)
        try:
            assert test_method(network, key) == result
        except ValueError:
            assert_array_equal(test_method(network, key), result)
            assert_units_equal(test_method(network, key), result)

    @pytest.mark.parametrize('key', keys, ids=key_names)
    @pytest.mark.parametrize('result', results, ids=result_ids)
    def test_saves_new_param_and_results_for_existing_analysis_key(self,
                                                                   network,
                                                                   key,
                                                                   result):
        test_method = make_test_method_with_key(result, ['test_key'])
        test_method(network, key)
        test_method(network, 2 * key)
        try:
            assert network.results['test'] == result
        except ValueError:
            assert_array_equal(network.results['test'], result)
            assert_units_equal(network.results['test'], result)

    def test_returns_existing_key_param_results_for_second_param(self,
                                                                 network):
        @lmt.utils._check_and_store(['test'], ['test_key'])
        def test_method(self, key):
            return key
        omegas = [10 * ureg.Hz, 11 * ureg.Hz]
        test_method(network, omegas[0])
        test_method(network, omegas[1])
        assert test_method(network, omegas[1]) == omegas[1]

    def test_result_not_calculated_twice_for_same_key(self,
                                                      mocker,
                                                      network):
        mock = mocker.Mock()
        
        @lmt.utils._check_and_store(['test'], ['key'])
        def test_function(network, key):
            return mock(key)
        
        test_function(network, 'key1')
        test_function(network, 'key1')
        mock.assert_called_once()

    def test_result_not_calculated_twice_for_second_key(self,
                                                        mocker,
                                                        network):
        mock = mocker.Mock()
        
        @lmt.utils._check_and_store(['test'], ['key'])
        def test_function(network, key):
            return mock(key)
        
        test_function(network, 10 * ureg.Hz)
        test_function(network, 11 * ureg.Hz)
        test_function(network, 11 * ureg.Hz)
        assert mock.call_count == 2

    def test_result_calculated_twice_for_differing_keys(self,
                                                        network):
        @lmt.utils._check_and_store(['test'], ['test_key'])
        def test_method(self, key):
            return key
        omegas = [10 * ureg.Hz, 11 * ureg.Hz]
        test_method(network, omegas[0])
        test_method(network, omegas[1])
        assert len(network.results_hash_dict) == 2
        
    def test_updates_results_and_analysis_params(self, network):
        
        @lmt.utils._check_and_store(['test'], ['test_key'])
        def test_method(self, key):
            return key
        
        omegas = [10 * ureg.Hz, 11 * ureg.Hz]
        test_method(network, omegas[0])
        results0 = network.results.copy()
        analysis_params0 = network.analysis_params.copy()
        test_method(network, omegas[1])
        results1 = network.results.copy()
        analysis_params1 = network.analysis_params.copy()
        test_method(network, omegas[0])
        results2 = network.results.copy()
        analysis_params2 = network.analysis_params.copy()
        check_quantity_dicts_are_equal(results0, results2)
        with pytest.raises(AssertionError):
            check_quantity_dicts_are_equal(results0, results1)
        check_quantity_dicts_are_equal(analysis_params0, analysis_params2)
        with pytest.raises(AssertionError):
            check_quantity_dicts_are_equal(analysis_params0, analysis_params1)
            
    def test_results_stored_in_result_dict_for_two_result_keys(
            self, network):
        @lmt.utils._check_and_store(['result1', 'result2'])
        def test_method(self):
            return 1 * ureg.mV, 2 * ureg.mV
        test_method(network)
        assert 1 * ureg.mV == network.results['result1']
        assert 2 * ureg.mV == network.results['result2']
        
    def test_results_stored_in_results_hash_dict_for_two_result_keys(
            self, network):
        @lmt.utils._check_and_store(['result1', 'result2'])
        def test_method(self):
            return 1 * ureg.mV, 2 * ureg.mV
        test_method(network)
        # get only element in results_hash_dict
        results_entry = list(network.results_hash_dict.values())[0]
        assert 1 * ureg.mV in results_entry.values()
        assert 2 * ureg.mV in results_entry.values()
        
    def test_results_stored_for_two_result_keys_with_one_analysis_key(
            self, network):
        @lmt.utils._check_and_store(['result1', 'result2'], ['analysis1'])
        def test_method(self, key):
            return 1 * ureg.mV, 2 * ureg.mV
        test_method(network, 1 * ureg.ms)
        assert 1 * ureg.mV == network.results['result1']
        assert 2 * ureg.mV == network.results['result2']
        assert 1 * ureg.ms == network.analysis_params['analysis1']
            
    def test_results_stored_for_two_result_keys_with_two_analysis_keys(
            self, network):
        @lmt.utils._check_and_store(['result1', 'result2'],
                                    ['analysis1', 'analysis2'])
        def test_method(self, key1, key2):
            return 1 * ureg.mV, 2 * ureg.mV
        test_method(network, 1 * ureg.ms, 2 * ureg.ms)
        assert 1 * ureg.mV == network.results['result1']
        assert 2 * ureg.mV == network.results['result2']
        assert 1 * ureg.ms == network.analysis_params['analysis1']
        assert 2 * ureg.ms == network.analysis_params['analysis2']
        
    def test_stores_in_results_hash_dict_two_result_keys_two_analysis_keys(
            self, network):
        @lmt.utils._check_and_store(['result1', 'result2'],
                                    ['analysis1', 'analysis2'])
        def test_method(self, key1, key2):
            return 1 * ureg.mV, 2 * ureg.mV
        test_method(network, 1 * ureg.ms, 2 * ureg.ms)
        # get only element in results_hash_dict
        results_entry = list(network.results_hash_dict.values())[0]
        assert 1 * ureg.mV in results_entry.values()
        assert 2 * ureg.mV in results_entry.values()
        assert 1 * ureg.ms in results_entry['analysis_params'].values()
        assert 2 * ureg.ms in results_entry['analysis_params'].values()
        
    def test_results_updated_in_result_dict_two_result_keys_one_analysis_key(
            self, network):
        @lmt.utils._check_and_store(['result1', 'result2'], ['analysis'])
        def test_method(self, key):
            return key * 1 * ureg.mV, key * 2 * ureg.mV
        test_method(network, 1)
        result11 = network.results['result1']
        result12 = network.results['result2']
        test_method(network, 2)
        result21 = network.results['result1']
        result22 = network.results['result2']
        assert 2 * result11 == result21
        assert 2 * result12 == result22
        
    def test_standard_value_for_analysis_key_is_not_passed(
            self, network):
        @lmt.utils._check_and_store(['result'], ['analysis'])
        def test_method(self, key=2):
            return key * 1 * ureg.mV
        test_method(network)
        assert network.analysis_params['analysis'] == 2
        assert network.results['result'] == 2 * ureg.mV
        
    def test_standard_value_for_analysis_key_is_passed_explicitely(
            self, network):
        @lmt.utils._check_and_store(['result'], ['analysis'])
        def test_method(self, key=2):
            return key * 1 * ureg.mV
        key = 5
        test_method(network, key=key)
        assert network.analysis_params['analysis'] == key
        assert network.results['result'] == key * ureg.mV
        
    def test_standard_values_for_two_analysis_keys_are_not_passed(
            self, network):
        @lmt.utils._check_and_store(['result'], ['analysis1', 'analysis2'])
        def test_method(self, key1=2, key2=3):
            return key1 * key2 * 1 * ureg.mV
        test_method(network)
        assert network.analysis_params['analysis1'] == 2
        assert network.analysis_params['analysis2'] == 3
        assert network.results['result'] == 6 * ureg.mV
        
    def test_standard_values_for_two_analysis_keys_are_passed_explicitely(
            self, network):
        @lmt.utils._check_and_store(['result'], ['analysis1', 'analysis2'])
        def test_method(self, key1=2, key2=3):
            return key1 * key2 * 1 * ureg.mV
        test_method(network, key1=3, key2=4)
        assert network.analysis_params['analysis1'] == 3
        assert network.analysis_params['analysis2'] == 4
        assert network.results['result'] == 12 * ureg.mV
            
    def test_only_first_standard_value_is_passed_explicitely(
            self, network):
        @lmt.utils._check_and_store(['result'], ['analysis1', 'analysis2'])
        def test_method(self, key1=2, key2=3):
            return key1 * key2 * 1 * ureg.mV
        test_method(network, key1=3)
        assert network.analysis_params['analysis1'] == 3
        assert network.analysis_params['analysis2'] == 3
        assert network.results['result'] == 9 * ureg.mV
            
    def test_only_second_standard_value_is_passed_explicitely(
            self, network):
        @lmt.utils._check_and_store(['result'], ['analysis1', 'analysis2'])
        def test_method(self, key1=2, key2=3):
            return key1 * key2 * 1 * ureg.mV
        test_method(network, key2=5)
        assert network.analysis_params['analysis1'] == 2
        assert network.analysis_params['analysis2'] == 5
        assert network.results['result'] == 10 * ureg.mV
        
    def test_one_positional_and_one_kwarg_with_std_args(
            self, network):
        @lmt.utils._check_and_store(['result'], ['analysis1', 'analysis2'])
        def test_method(self, key1=2, key2=3):
            return key1 * key2 * 1 * ureg.mV
        test_method(network, 3, key2=5)
        assert network.analysis_params['analysis1'] == 3
        assert network.analysis_params['analysis2'] == 5
        assert network.results['result'] == 15 * ureg.mV
            
    def test_one_and_two_are_passed_with_three_std_args(
            self, network):
        @lmt.utils._check_and_store(['result'], ['analysis1',
                                                 'analysis2',
                                                 'analysis3'])
        def test_method(self, key1=1, key2=2, key3=3):
            return key1 * key2 * key3 * ureg.mV
        test_method(network, 3, 4)
        assert network.analysis_params['analysis1'] == 3
        assert network.analysis_params['analysis2'] == 4
        assert network.analysis_params['analysis3'] == 3
        assert network.results['result'] == 36 * ureg.mV
    
    def test_one_and_three_are_passed_with_three_mixed_std_args(
            self, network):
        @lmt.utils._check_and_store(['result'], ['analysis1',
                                                 'analysis2',
                                                 'analysis3'])
        def test_method(self, key1, key2=2, key3=3):
            return key1 * key2 * key3 * ureg.mV
        test_method(network, 3, key3=5)
        assert network.analysis_params['analysis1'] == 3
        assert network.analysis_params['analysis2'] == 2
        assert network.analysis_params['analysis3'] == 5
        assert network.results['result'] == 30 * ureg.mV
            
    def test_one_and_three_are_passed_with_std_args(
            self, network):
        @lmt.utils._check_and_store(['result'], ['analysis1',
                                                 'analysis2',
                                                 'analysis3'])
        def test_method(self, key1=1, key2=2, key3=3):
            return key1 * key2 * key3 * ureg.mV
        test_method(network, 3, key3=5)
        assert network.analysis_params['analysis1'] == 3
        assert network.analysis_params['analysis2'] == 2
        assert network.analysis_params['analysis3'] == 5
        assert network.results['result'] == 30 * ureg.mV
