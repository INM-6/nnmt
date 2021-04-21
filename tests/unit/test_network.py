import re
import pytest
import numpy as np
from numpy.testing import assert_allclose

from ..checks import (assert_array_equal,
                      assert_units_equal,
                      check_quantity_dicts_are_equal)

import lif_meanfield_tools as lmt

ureg = lmt.ureg


class Test_initialization:

    @pytest.mark.parametrize('key, value', [('tau_m', 10 * ureg.ms),
                                            ('d_i_sd', 0.375 * ureg.ms),
                                            ('label', 'microcircuit')])
    def test_correct_network_params_loaded(self, network, key, value):
        assert network.network_params[key] == value

    @pytest.mark.parametrize('key, value', [('f_min', 0.1 * ureg.Hz),
                                            ('omega', 20 * ureg.Hz),
                                            ('k_max', 100.5 / ureg.mm)])
    def test_correct_analysis_params_loaded(self, network, key, value):
        assert network.analysis_params[key] == value

    def test_network_with_given_network_params_created(self):
        network_params = dict(tau_m=10 * ureg.ms,
                              tau_s=5 * ureg.ms)
        network = lmt.Network(new_network_params=network_params,
                              derive_params=False)
        assert network.network_params == network_params

    def test_network_with_given_analysis_params_created(self):
        analysis_params = dict(f_min=1 * ureg.Hz,
                               f_max=10 * ureg.Hz)
        network = lmt.Network(new_analysis_params=analysis_params,
                              derive_params=False)
        assert network.analysis_params == analysis_params

    def test_network_params_updated_on_initialization(self,
                                                      network_params_yaml,
                                                      analysis_params_yaml):
        tau_m = 1000 * ureg.ms
        network = lmt.Network(network_params_yaml,
                              analysis_params_yaml,
                              new_network_params=dict(tau_m=tau_m))
        assert network.network_params['tau_m'] == tau_m

    def test_analysis_params_updated_on_initialization(self,
                                                       network_params_yaml,
                                                       analysis_params_yaml):
        df = 1000 * ureg.Hz
        network = lmt.Network(network_params_yaml,
                              analysis_params_yaml,
                              new_analysis_params=dict(df=df))
        assert network.analysis_params['df'] == df

    @pytest.mark.xfail
    def test_warning_is_given_if_necessary_parameters_are_missing(self):
        """What are necessary parameters? For what?"""
        raise NotImplementedError

    def test_if_derive_params_false_no_calculation_of_derived_params(
            self, mocker, network_params_yaml, analysis_params_yaml):
        mocker.patch.object(lmt.Network,
                            '_calculate_dependent_network_parameters',
                            autospec=True)
        mocker.patch.object(lmt.Network,
                            '_calculate_dependent_analysis_parameters',
                            autospec=True)
        network = lmt.Network(network_params_yaml,
                              analysis_params_yaml,
                              derive_params=False)
        network._calculate_dependent_network_parameters.assert_not_called()
        network._calculate_dependent_analysis_parameters.assert_not_called()

    def test_hash_is_created(self, network):
        assert hasattr(network, 'hash')

    def test_loading_of_existing_results(self, unit_fixture_path):
        network = lmt.Network(file=f'{unit_fixture_path}test_network.h5')
        assert len(network.network_params.items()) != 0
        assert len(network.analysis_params.items()) != 0
        assert 'firing_rates' in network.results.keys()

    def test_result_dict_is_created(self, network):
        assert hasattr(network, 'results')

    def test_results_hash_dict_is_created(self, network):
        assert hasattr(network, 'results_hash_dict')
    
    
class Test_calculation_of_dependent_network_params:
    """
    Depends strongly on network_params_microcircuit.yaml in
    tests/fixtures/unit/config/.
    """

    def test_dimension(self, network):
        assert network.network_params['dimension'] == 8

    def test_V0_rel(self, network):
        assert network.network_params['V_0_rel'] == 0 * ureg.mV

    def test_V_th_rel(self, network):
        assert network.network_params['V_th_rel'] == 15 * ureg.mV

    def test_j(self, network):
        assert network.network_params['j'] == 0.1756 * ureg.mV

    def test_W(self, network):
        W = [[87.8, -351.2, 87.8, -351.2, 87.8, -351.2, 87.8, -351.2]
             for i in range(network.network_params['dimension'])] * ureg.pA
        W[0][2] *= 2
        assert_array_equal(network.network_params['W'], W)
        assert_units_equal(network.network_params['W'], W)

    def test_J(self, network):
        J = [[0.1756, -0.7024, 0.1756, -0.7024, 0.1756, -0.7024, 0.1756,
              -0.7024] for i in range(network.network_params['dimension'])
             ] * ureg.mV
        J[0][2] *= 2
        assert_array_equal(network.network_params['J'], J)
        assert_units_equal(network.network_params['J'], J)

    def test_Delay(self, network):
        Delay = [[1.5, 0.75, 1.5, 0.75, 1.5, 0.75, 1.5, 0.75]
                 for i in range(network.network_params['dimension'])
                 ] * ureg.ms
        assert_array_equal(network.network_params['Delay'], Delay)
        assert_units_equal(network.network_params['Delay'], Delay)

    def test_Delay_sd(self, network):
        Delay_sd = [[0.75, 0.375, 0.75, 0.375, 0.75, 0.375, 0.75, 0.375]
                    for i in range(network.network_params['dimension'])
                    ] * ureg.ms
        assert_array_equal(network.network_params['Delay_sd'], Delay_sd)
        assert_units_equal(network.network_params['Delay_sd'], Delay_sd)


class Test_calculation_of_dependent_analysis_params:
    """Depends strongly on analysis_params_test.yaml in tests/fixtures."""

    def test_omegas(self, network):
        omegas = [6.28318531e-01,
                  1.89123878e+02,
                  3.77619437e+02,
                  5.66114996e+02,
                  7.54610555e+02,
                  9.43106115e+02,
                  1.13160167e+03,
                  1.32009723e+03,
                  1.50859279e+03,
                  1.69708835e+03] * ureg.Hz
        assert_allclose(network.analysis_params['omegas'].magnitude,
                        omegas.magnitude, 1e-5)
        assert_units_equal(network.analysis_params['omegas'].magnitude,
                           omegas.magnitude)

    def test_k_wavenumbers(self, network):
        k_wavenumbers = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91] * (1 / ureg.mm)
        assert_array_equal(network.analysis_params['k_wavenumbers'],
                           k_wavenumbers)
        assert_units_equal(network.analysis_params['k_wavenumbers'],
                           k_wavenumbers)


class Test_saving_and_loading:
    
    def test_save_creates_h5_file(self, mocker, tmpdir, network):
        # create mocking method
        test_method = make_test_method(42)
        mocker.patch.object(lmt.Network, 'mean_input', new=test_method)
        # create temp directory
        tmp_test = tmpdir.mkdir('tmp_test')
        # calculate and save mean_input in tmp dir
        with tmp_test.as_cwd():
            network.mean_input()
            network.save('test.h5')
        # results file name expression
        exp = re.compile(r'.*\.h5')
        # file names in tmp dir
        files = [str(obj) for obj in tmp_test.listdir()]
        # file names matching exp
        matches = list(filter(exp.match, files))
        # pass test if test file created
        assert any(matches)

    def test_save_created_output_file_with_results(
            self, mocker, tmpdir, network):
        # create mocking method
        test_method = make_test_method(42)
        mocker.patch.object(lmt.Network, 'mean_input', new=test_method)
        network.mean_input()
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            network.save('test.h5')
            output = lmt.input_output.load_val_unit_dict_from_h5('test.h5')
            assert 'test' in output['results'].keys()

    def test_save_overwriting_existing_file_raises_error(self, mocker, tmpdir,
                                                         network):
        # create mocking method
        test_method = make_test_method(42)
        mocker.patch.object(lmt.Network, 'mean_input', new=test_method)
        file = 'file.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            with pytest.raises(IOError):
                network.mean_input()
                network.save(file=file)
                network.save(file=file)

    def test_save_overwrites_existing_file_if_explicitely_told(
            self, tmpdir, mocker, network):
        file = 'file.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            test_method = make_test_method(42)
            mocker.patch.object(lmt.Network, 'mean_input', new=test_method)
            network.mean_input()
            network.save(file=file)
            test_method = make_test_method(43)
            mocker.patch.object(lmt.Network, 'mean_input', new=test_method)
            new_mean = network.mean_input()
            network.save(file=file, overwrite=True)
            output = lmt.input_output.load_val_unit_dict_from_h5(file)
            assert_array_equal(output['results']['test'], new_mean)
            
    def test_load_correctly_sets_network_dictionaries(self, tmpdir, network):
        network.firing_rates()
        nparams = network.network_params
        aparams = network.analysis_params
        results = network.results
        rhd = network.results_hash_dict
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            network.save('test.h5')
            network.network_params = {}
            network.analysis_params = {}
            network.results = {}
            network.results_hash_dict = {}
            network.load('test.h5')
            check_quantity_dicts_are_equal(nparams, network.network_params)
            check_quantity_dicts_are_equal(aparams, network.analysis_params)
            check_quantity_dicts_are_equal(results, network.results)
            check_quantity_dicts_are_equal(rhd, network.results_hash_dict)
            

class Test_meta_functions:

    def test_show(self, mocker, network):
        assert network.show() == []
        
        @lmt.Network._check_and_store(['inner'])
        def test_method_inner(self):
            return 1
        
        @lmt.Network._check_and_store(['outer'])
        def test_method_outer(self):
            return test_method_inner(self)
        
        mocker.patch.object(lmt.Network, 'mean_input', new=test_method_outer)
        network.mean_input()
        assert network.show() == ['inner', 'outer']

    def test_change_network_parameters(self, network):
        new_tau_m = 1000 * ureg.ms
        update = dict(tau_m=new_tau_m)
        network.change_parameters(changed_network_params=update,
                                  overwrite=True)
        assert network.network_params['tau_m'] == new_tau_m

    def test_change_analysis_parameters(self, network):
        new_df = 1000 * ureg.Hz
        update = dict(df=new_df)
        network.change_parameters(changed_analysis_params=update,
                                  overwrite=True)
        assert network.analysis_params['df'] == new_df
        
    def test_change_parameters_returns_new_network(self, network):
        new_tau_m = 1000 * ureg.ms
        update = dict(tau_m=new_tau_m)
        new_network = network.change_parameters(changed_network_params=update)
        assert new_network is not network
        
    def test_change_parameters_returns_new_network_with_uncoupled_dicts(
            self, network):
        new_tau_m = 1000 * ureg.ms
        update = dict(tau_m=new_tau_m)
        new_network = network.change_parameters(changed_network_params=update)
        new_network.network_params['K'] = np.array([1, 2, 3])
        new_network.analysis_params['omegas'] = np.array([1, 2, 3]) * ureg.Hz
        with pytest.raises(AssertionError):
            assert_array_equal(network.network_params['K'],
                               new_network.network_params['K'])
        with pytest.raises(AssertionError):
            assert_array_equal(network.analysis_params['omegas'],
                               new_network.analysis_params['omegas'])

    def test_change_parameter_deletes_results_if_overwrite_true(self, network):
        new_tau_m = 1000 * ureg.ms
        update = dict(tau_m=new_tau_m)
        network.change_parameters(changed_network_params=update,
                                  overwrite=True)
        assert len(network.results.items()) == 0
        assert len(network.results_hash_dict.items()) == 0

    @pytest.mark.xfail
    def test_extend_analysis_frequencies(self):
        raise NotImplementedError
    

def make_test_method(output):
    @lmt.Network._check_and_store(['test'])
    def test_method(self):
        return output
    return test_method


def make_test_method_with_key(output, key):
    @lmt.Network._check_and_store(['test'], key)
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
    def test_save_results(self, mocker, network, test_method, result):
        mocker.patch.object(lmt.Network, 'mean_input', new=test_method)
        network.mean_input()
        try:
            assert network.results['test'] == result
        except ValueError:
            assert_array_equal(network.results['test'], result)
            assert_units_equal(network.results['test'], result)

    @pytest.mark.parametrize('test_method, result', zip(test_methods, results),
                             ids=result_ids)
    def test_returns_existing_result(self, mocker, network, test_method,
                                     result):
        mocker.patch.object(lmt.Network, 'mean_input', new=test_method)
        network.mean_input()
        try:
            assert network.mean_input() == result
        except ValueError:
            assert_array_equal(network.mean_input(), result)
            assert_units_equal(network.mean_input(), result)

    def test_result_not_calculated_twice(self, mocker, network):
        mocked = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                              'firing_rates')
        network.firing_rates()
        network.firing_rates()
        mocked.assert_called_once()

    @pytest.mark.parametrize('key', keys, ids=key_names)
    @pytest.mark.parametrize('result', results, ids=result_ids)
    def test_saves_new_analysis_key_with_param_and_results(self,
                                                           mocker,
                                                           network,
                                                           key,
                                                           result):
        test_method = make_test_method_with_key(result, ['test_key'])
        mocker.patch.object(lmt.Network, 'mean_input', new=test_method)
        network.mean_input(key)
        try:
            assert network.results['test'] == result
        except ValueError:
            assert_array_equal(network.results['test'], result)
            assert_units_equal(network.results['test'], result)

    @pytest.mark.parametrize('key', keys, ids=key_names)
    @pytest.mark.parametrize('result', results, ids=result_ids)
    def test_returns_existing_analysis_key_with_param_and_results(self,
                                                                  mocker,
                                                                  network,
                                                                  key,
                                                                  result):
        test_method = make_test_method_with_key(result, ['test_key'])
        mocker.patch.object(lmt.Network, 'mean_input', new=test_method)
        network.mean_input(key)
        try:
            assert network.mean_input(key) == result
        except ValueError:
            assert_array_equal(network.mean_input(key), result)
            assert_units_equal(network.mean_input(key), result)

    @pytest.mark.parametrize('key', keys, ids=key_names)
    @pytest.mark.parametrize('result', results, ids=result_ids)
    def test_saves_new_param_and_results_for_existing_analysis_key(self,
                                                                   mocker,
                                                                   network,
                                                                   key,
                                                                   result):
        test_method = make_test_method_with_key(result, ['test_key'])
        mocker.patch.object(lmt.Network, 'mean_input', new=test_method)
        network.mean_input(key)
        network.mean_input(2 * key)
        try:
            assert network.results['test'] == result
        except ValueError:
            assert_array_equal(network.results['test'], result)
            assert_units_equal(network.results['test'], result)

    def test_returns_existing_key_param_results_for_second_param(self,
                                                                 mocker,
                                                                 network):
        @lmt.Network._check_and_store(['test'], ['test_key'])
        def test_method(self, key):
            return key
        mocker.patch('lif_meanfield_tools.Network.mean_input', new=test_method)
        omegas = [10 * ureg.Hz, 11 * ureg.Hz]
        network.mean_input(omegas[0])
        network.mean_input(omegas[1])
        assert network.mean_input(omegas[1]) == omegas[1]

    def test_result_not_calculated_twice_for_same_key(self,
                                                      mocker,
                                                      network):
        mocked = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                              'delay_dist_matrix')
        network.delay_dist_matrix(10 * ureg.Hz)
        network.delay_dist_matrix(10 * ureg.Hz)
        mocked.assert_called_once()

    def test_result_not_calculated_twice_for_second_key(self,
                                                        mocker,
                                                        network):
        mocked = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                              'delay_dist_matrix')
        network.delay_dist_matrix(10 * ureg.Hz)
        network.delay_dist_matrix(11 * ureg.Hz)
        network.delay_dist_matrix(11 * ureg.Hz)
        assert mocked.call_count == 2

    def test_result_calculated_twice_for_differing_keys(self,
                                                        mocker,
                                                        network):
        @lmt.Network._check_and_store(['test'], ['test_key'])
        def test_method(self, key):
            return key
        mocker.patch('lif_meanfield_tools.Network.mean_input', new=test_method)
        omegas = [10 * ureg.Hz, 11 * ureg.Hz]
        network.mean_input(omegas[0])
        network.mean_input(omegas[1])
        assert len(network.results_hash_dict) == 2
        
    def test_updates_results_and_analysis_params(self, mocker, network):
        @lmt.Network._check_and_store(['test'], ['test_key'])
        def test_method(self, key):
            return key
        mocker.patch('lif_meanfield_tools.Network.mean_input', new=test_method)
        omegas = [10 * ureg.Hz, 11 * ureg.Hz]
        network.mean_input(omegas[0])
        results0 = network.results.copy()
        analysis_params0 = network.analysis_params.copy()
        network.mean_input(omegas[1])
        results1 = network.results.copy()
        analysis_params1 = network.analysis_params.copy()
        network.mean_input(omegas[0])
        results2 = network.results.copy()
        analysis_params2 = network.analysis_params.copy()
        check_quantity_dicts_are_equal(results0, results2)
        with pytest.raises(AssertionError):
            check_quantity_dicts_are_equal(results0, results1)
        check_quantity_dicts_are_equal(analysis_params0, analysis_params2)
        with pytest.raises(AssertionError):
            check_quantity_dicts_are_equal(analysis_params0, analysis_params1)
            
    def test_results_stored_in_result_dict_for_two_result_keys(
            self, mocker, network):
        @lmt.Network._check_and_store(['result1', 'result2'])
        def test_method(self):
            return 1 * ureg.mV, 2 * ureg.mV
        mocker.patch('lif_meanfield_tools.Network.mean_input', new=test_method)
        network.mean_input()
        assert 1 * ureg.mV == network.results['result1']
        assert 2 * ureg.mV == network.results['result2']
        
    def test_results_stored_in_results_hash_dict_for_two_result_keys(
            self, mocker, network):
        @lmt.Network._check_and_store(['result1', 'result2'])
        def test_method(self):
            return 1 * ureg.mV, 2 * ureg.mV
        mocker.patch('lif_meanfield_tools.Network.mean_input', new=test_method)
        network.mean_input()
        # get only element in results_hash_dict
        results_entry = list(network.results_hash_dict.values())[0]
        assert 1 * ureg.mV in results_entry.values()
        assert 2 * ureg.mV in results_entry.values()
        
    def test_results_stored_for_two_result_keys_with_one_analysis_key(
            self, mocker, network):
        @lmt.Network._check_and_store(['result1', 'result2'], ['analysis1'])
        def test_method(self, key):
            return 1 * ureg.mV, 2 * ureg.mV
        mocker.patch('lif_meanfield_tools.Network.mean_input', new=test_method)
        network.mean_input(1 * ureg.ms)
        assert 1 * ureg.mV == network.results['result1']
        assert 2 * ureg.mV == network.results['result2']
        assert 1 * ureg.ms == network.analysis_params['analysis1']
            
    def test_results_stored_for_two_result_keys_with_two_analysis_keys(
            self, mocker, network):
        @lmt.Network._check_and_store(['result1', 'result2'],
                                      ['analysis1', 'analysis2'])
        def test_method(self, key1, key2):
            return 1 * ureg.mV, 2 * ureg.mV
        mocker.patch('lif_meanfield_tools.Network.mean_input', new=test_method)
        network.mean_input(1 * ureg.ms, 2 * ureg.ms)
        assert 1 * ureg.mV == network.results['result1']
        assert 2 * ureg.mV == network.results['result2']
        assert 1 * ureg.ms == network.analysis_params['analysis1']
        assert 2 * ureg.ms == network.analysis_params['analysis2']
        
    def test_stores_in_results_hash_dict_two_result_keys_two_analysis_keys(
            self, mocker, network):
        @lmt.Network._check_and_store(['result1', 'result2'],
                                      ['analysis1', 'analysis2'])
        def test_method(self, key1, key2):
            return 1 * ureg.mV, 2 * ureg.mV
        mocker.patch('lif_meanfield_tools.Network.mean_input', new=test_method)
        network.mean_input(1 * ureg.ms, 2 * ureg.ms)
        # get only element in results_hash_dict
        results_entry = list(network.results_hash_dict.values())[0]
        assert 1 * ureg.mV in results_entry.values()
        assert 2 * ureg.mV in results_entry.values()
        assert 1 * ureg.ms in results_entry['analysis_params'].values()
        assert 2 * ureg.ms in results_entry['analysis_params'].values()
        
    def test_results_updated_in_result_dict_two_result_keys_one_analysis_key(
            self, mocker, network):
        @lmt.Network._check_and_store(['result1', 'result2'], ['analysis'])
        def test_method(self, key):
            return key * 1 * ureg.mV, key * 2 * ureg.mV
        mocker.patch('lif_meanfield_tools.Network.mean_input', new=test_method)
        network.mean_input(1)
        result11 = network.results['result1']
        result12 = network.results['result2']
        network.mean_input(2)
        result21 = network.results['result1']
        result22 = network.results['result2']
        assert 2 * result11 == result21
        assert 2 * result12 == result22
        
            
class Test_functionality:
    """A lot of those tests might actually be superfluous."""

    def test_firing_rates_calls_correctly(self, network, mocker):
        mock = mocker.patch('lif_meanfield_tools.meanfield_calcs.firing_rates')
        network.firing_rates()
        mock.assert_called_once()

    def test_mean_input_calls_correctly(self, network, mocker):
        mock_mean = mocker.patch('lif_meanfield_tools.meanfield_calcs.mean')
        mock_fr = mocker.patch('lif_meanfield_tools.Network.firing_rates')
        network.mean_input()
        mock_mean.assert_called_once()
        mock_fr.assert_called_once()

    def test_std_input_calls_correctly(self, network, mocker):
        mock_std = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                                'standard_deviation')
        mock_fr = mocker.patch('lif_meanfield_tools.Network.firing_rates')
        network.std_input()
        mock_std.assert_called_once()
        mock_fr.assert_called_once()

    def test_working_point_calls_calls_correctly(self, network, mocker):
        mock_fr = mocker.patch('lif_meanfield_tools.Network.firing_rates')
        mock_mean = mocker.patch('lif_meanfield_tools.Network.mean_input')
        mock_std = mocker.patch('lif_meanfield_tools.Network.std_input')
        network.working_point()
        mock_mean.assert_called_once()
        mock_std.assert_called_once()
        mock_fr.assert_called_once()

    def test_delay_dist_matrix_calls_delay_dist_matrix_multi(self,
                                                             network,
                                                             mocker):
        mock = mocker.patch('lif_meanfield_tools.Network.'
                            'delay_dist_matrix_multi')
        network.delay_dist_matrix()
        mock.assert_called_once()

    def test_delay_dist_matrix_calls_delay_dist_matrix_single(self,
                                                              network,
                                                              mocker):
        mock = mocker.patch('lif_meanfield_tools.Network.'
                            'delay_dist_matrix_single')
        network.delay_dist_matrix(1 * ureg.Hz)
        mock.assert_called_once()

    def test_delay_dist_matrix_multi_calls_correctly(self, network, mocker):
        mock = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                            'delay_dist_matrix')
        network.delay_dist_matrix_multi()
        mock.assert_called_once()

    def test_delay_dist_matrix_single_calls_correctly(self, network, mocker):
        mock = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                            'delay_dist_matrix')
        network.delay_dist_matrix_single(1 * ureg.Hz)
        mock.assert_called_once()

    def test_transfer_function_calls_transfer_function_multi(self,
                                                             network,
                                                             mocker):
        mock = mocker.patch('lif_meanfield_tools.Network.'
                            'transfer_function_multi')
        network.transfer_function()
        mock.assert_called_once()

    def test_transfer_function_calls_transfer_function_single(self,
                                                              network,
                                                              mocker):
        mock = mocker.patch('lif_meanfield_tools.Network.'
                            'transfer_function_single')
        network.transfer_function(1 * ureg.Hz)
        mock.assert_called_once()

    def test_transfer_function_multi_calls_correctly(self, network, mocker):
        mock_tf = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                               'transfer_function')
        mock_mean = mocker.patch('lif_meanfield_tools.Network.mean_input')
        mock_std = mocker.patch('lif_meanfield_tools.Network.std_input')
        network.transfer_function_multi()
        mock_tf.assert_called_once()
        mock_mean.assert_called_once()
        mock_std.assert_called_once()

    def test_transfer_function_single_calls_correctly(self, network, mocker):
        mock_tf = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                               'transfer_function')
        mock_mean = mocker.patch('lif_meanfield_tools.Network.mean_input')
        mock_std = mocker.patch('lif_meanfield_tools.Network.std_input')
        network.transfer_function_single(1 * ureg.Hz)
        mock_tf.assert_called_once()
        mock_mean.assert_called_once()
        mock_std.assert_called_once()

    def test_sensitivity_measure_calls_correctly(self, network, mocker):
        mock_mean = mocker.patch('lif_meanfield_tools.Network.mean_input')
        mock_std = mocker.patch('lif_meanfield_tools.Network.std_input')
        mock_sm = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                               'sensitivity_measure')
        mock_tf = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                               'transfer_function')
        mock_dd = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                               'delay_dist_matrix')
        network.sensitivity_measure(1 * ureg.Hz)
        mock_mean.assert_called_once()
        mock_std.assert_called_once()
        mock_sm.assert_called_once()
        mock_tf.assert_called_once()
        mock_dd.assert_called_once()

    def test_transfer_function_is_conjugated_if_omega_negative(self, network,
                                                               mocker):
        mocker.patch('lif_meanfield_tools.Network.mean_input')
        mocker.patch('lif_meanfield_tools.Network.std_input')
        mock_sm = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                               'sensitivity_measure')
        mock_tf = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                               'transfer_function')
        mocker.patch('lif_meanfield_tools.meanfield_calcs.delay_dist_matrix')
        tf = np.array([[complex(1, 2), complex(3, 4)],
                       [complex(5, 6), complex(7, 8)]])
        mock_tf.return_value = tf
        network.sensitivity_measure(- 1 * ureg.Hz)
        assert_array_equal(mock_sm.call_args[0][0], np.conjugate(tf))

    def test_power_spectra_calls_correctly(self, network, mocker):
        mock_ps = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                               'power_spectra')
        mock_fr = mocker.patch('lif_meanfield_tools.Network.firing_rates')
        mock_dd = mocker.patch('lif_meanfield_tools.Network.delay_dist_matrix')
        mock_tf = mocker.patch('lif_meanfield_tools.Network.transfer_function')
        network.power_spectra()
        mock_ps.assert_called_once()
        mock_fr.assert_called_once()
        mock_dd.assert_called_once()
        mock_tf.assert_called_once()

    def test_eigenvalue_spectra_calls_correctly(self, network, mocker):
        mock_es = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                               'eigen_spectra')
        mock_dd = mocker.patch('lif_meanfield_tools.Network.delay_dist_matrix')
        mock_tf = mocker.patch('lif_meanfield_tools.Network.transfer_function')
        network.eigenvalue_spectra('MH')
        mock_es.assert_called_once()
        mock_dd.assert_called_once()
        mock_tf.assert_called_once()

    def test_r_eigenvec_spectra_calls_correctly(self, network, mocker):
        mock_es = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                               'eigen_spectra')
        mock_dd = mocker.patch('lif_meanfield_tools.Network.delay_dist_matrix')
        mock_tf = mocker.patch('lif_meanfield_tools.Network.transfer_function')
        network.r_eigenvec_spectra('MH')
        mock_es.assert_called_once()
        mock_dd.assert_called_once()
        mock_tf.assert_called_once()

    def test_l_eigenvec_spectra_calls_correctly(self, network, mocker):
        mock_es = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                               'eigen_spectra')
        mock_dd = mocker.patch('lif_meanfield_tools.Network.delay_dist_matrix')
        mock_tf = mocker.patch('lif_meanfield_tools.Network.transfer_function')
        network.l_eigenvec_spectra('MH')
        mock_es.assert_called_once()
        mock_dd.assert_called_once()
        mock_tf.assert_called_once()

    def test_additional_rates_for_fixed_input_calls_correctly(self, network,
                                                              mocker):
        mock = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                            'additional_rates_for_fixed_input')
        mock.return_value = 1, 2
        network.additional_rates_for_fixed_input(1 * ureg.Hz, 2 * ureg.Hz)
        mock.assert_called_once()

    def test_fit_transfer_function_calls_correctly(self, network, mocker):
        mock = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                            'fit_transfer_function')
        mock.return_value = 1, 2, 3, 4
        mock_ecs = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                                'effective_coupling_strength')
        mock_ecs.return_value = 1
        mock_tf = mocker.patch('lif_meanfield_tools.Network.transfer_function')
        mock_mean = mocker.patch('lif_meanfield_tools.Network.mean_input')
        mock_std = mocker.patch('lif_meanfield_tools.Network.std_input')
        network.fit_transfer_function()
        mock.assert_called_once()
        mock_ecs.assert_called_once()
        mock_tf.assert_called_once()
        mock_mean.assert_called_once()
        mock_std.assert_called_once()

    def test_scan_fit_transfer_function_mean_std_input_calls_correctly(self,
                                                                       network,
                                                                       mocker):
        mock = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                            'scan_fit_transfer_function_mean_std_input')
        mock.return_value = 1, 2
        network.scan_fit_transfer_function_mean_std_input(1, 2)
        mock.assert_called_once()

    def test_linear_interpolation_alpha_called_correctly(self, network,
                                                         mocker):
        mock = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                            'linear_interpolation_alpha')
        mock.return_value = 1, 2, 3, 4, 5, 6
        mock_mean = mocker.patch('lif_meanfield_tools.Network.mean_input')
        mock_std = mocker.patch('lif_meanfield_tools.Network.std_input')
        network.linear_interpolation_alpha(1, 2)
        mock.assert_called_once()
        mock_mean.assert_called_once()
        mock_std.assert_called_once()
