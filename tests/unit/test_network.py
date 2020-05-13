import re
import pytest
import numpy as np

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
    
    @pytest.mark.xfail
    def test_loading_of_existing_results(self):
        """How do we want them to be loaded?"""
        raise NotImplementedError
    
    def test_result_dict_is_created(self, network):
        assert hasattr(network, 'results')
    

class Test_calculation_of_dependent_network_params:
    """
    Depends strongly on network_params_microcircuit.yaml in tests/fixtures.
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
        np.testing.assert_array_equal(network.network_params['W'], W)
    
    def test_J(self, network):
        J = [[0.1756, -0.7024, 0.1756, -0.7024, 0.1756, -0.7024, 0.1756,
              -0.7024] for i in range(network.network_params['dimension'])
             ] * ureg.mV
        J[0][2] *= 2
        np.testing.assert_array_equal(network.network_params['J'], J)

    def test_Delay(self, network):
        Delay = [[1.5, 0.75, 1.5, 0.75, 1.5, 0.75, 1.5, 0.75]
                 for i in range(network.network_params['dimension'])
                 ] * ureg.ms
        np.testing.assert_array_equal(network.network_params['Delay'], Delay)

    def test_Delay_sd(self, network):
        Delay_sd = [[0.75, 0.375, 0.75, 0.375, 0.75, 0.375, 0.75, 0.375]
                    for i in range(network.network_params['dimension'])
                    ] * ureg.ms
        np.testing.assert_array_equal(network.network_params['Delay_sd'],
                                      Delay_sd)
    

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
        np.testing.assert_allclose(network.analysis_params['omegas'].magnitude,
                                   omegas.magnitude, 1e-5)
        assert network.analysis_params['omegas'].units == omegas.units
    
    def test_k_wavenumbers(self, network):
        k_wavenumbers = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91] * (1 / ureg.mm)
        np.testing.assert_array_equal(network.analysis_params['k_wavenumbers'],
                                      k_wavenumbers)
    

class Test_saving_and_loading:
    
    def test_save_creates_h5_file(self, tmpdir, network):
        # create temp directory
        tmp_test = tmpdir.mkdir('tmp_test')
        # calculate and save mean_input in tmp dir
        with tmp_test.as_cwd():
            network.mean_input()
            network.save()
        # results file name expression
        exp = re.compile(r'.*microcircuit.*\.h5')
        # file names in tmp dir
        file_names = [str(obj) for obj in tmp_test.listdir()]
        # file names matching exp
        matches = list(filter(exp.match, file_names))
        # pass test if test file created
        assert any(matches)
                
    def test_save_with_given_file_name(self, tmpdir, network):
        file_name = 'my_file_name.h5'
        # create temp directory
        tmp_test = tmpdir.mkdir('tmp_test')
        # calculate and save mean_input in tmp dir
        with tmp_test.as_cwd():
            network.mean_input()
            network.save(file_name=file_name)
        # results file name expression
        exp = re.compile(r'.*{}'.format(file_name))
        # file names in tmp dir
        file_names = [str(obj) for obj in tmp_test.listdir()]
        # file names matching exp
        matches = list(filter(exp.match, file_names))
        # pass test if test file created
        assert any(matches)
    
    @pytest.mark.improve
    def test_save_passed_output(self, tmpdir, network):
        """
        This checks if some file is created, but doesn't check its content!
        Improve as soon as possible!
        """
        file_name = 'file.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        output_key = 'my_key'
        output = dict(my_var=10 * ureg.m)
        with tmp_test.as_cwd():
            network.save(output_key=output_key,
                         output=output,
                         file_name=file_name)
        # results file name expression
        exp = re.compile(r'.*{}'.format(file_name))
        # file names in tmp dir
        file_names = [str(obj) for obj in tmp_test.listdir()]
        # file names matching exp
        matches = list(filter(exp.match, file_names))
        # pass test if test file created
        assert any(matches)
        
    @pytest.mark.xfail
    def test_save_passed_output_without_key_raises_exception(self,
                                                             tmpdir,
                                                             network):
        file_name = 'file.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        output = dict(my_var=10 * ureg.m)
        with tmp_test.as_cwd():
            with pytest.raises(IOError):
                network.save(output=output, file_name=file_name)
        
    @pytest.mark.xfail
    def test_save_passed_output_without_output_raises_exception(self,
                                                                tmpdir,
                                                                network):
        file_name = 'file.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        output_key = 'my_key'
        with tmp_test.as_cwd():
            with pytest.raises(IOError):
                network.save(output_key=output_key, file_name=file_name)
    
    @pytest.mark.xfail
    def test_save_overwriting_existing_file_raises_error(self, tmpdir,
                                                         network):
        file_name = 'file.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            with pytest.raises(IOError):
                network.mean_input()
                network.save(file_name=file_name)
                network.save(file_name=file_name)
                
    @pytest.mark.improve
    @pytest.mark.xfail
    def test_save_overwrites_existing_file_if_explicitely_told(self, tmpdir,
                                                               network):
        file_name = 'file.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            network.mean_input()
            network.save(file_name=file_name)
            network.network_params['tau_m'] = 100 * ureg.ms
            new_mean = network.mean_input()
            network.save(file_name=file_name, overwrite=True)
            network.load(file_name=file_name)
            assert network.results['mean_input'] == new_mean
    
    
class Test_meta_functions:
        
    def test_show(self, network):
        assert network.show() == []
        network.mean_input()
        assert network.show() == ['firing_rates', 'mean_input']

    def test_change_network_parameters(self, network):
        new_tau_m = 1000 * ureg.ms
        update = dict(tau_m=new_tau_m)
        network.change_parameters(changed_network_params=update)
        assert network.network_params['tau_m'] == new_tau_m

    def test_change_analysis_parameters(self, network):
        new_df = 1000 * ureg.Hz
        update = dict(df=new_df)
        network.change_parameters(changed_analysis_params=update)
        assert network.analysis_params['df'] == new_df
    
    @pytest.mark.xfail
    def test_extend_analysis_frequencies(self):
        raise NotImplementedError


def make_test_method(output):
    @lmt.Network._check_and_store('test')
    def test_method(self):
        return output
    return test_method


def make_test_method_with_key(output, key):
    @lmt.Network._check_and_store('test', key)
    def test_method(self, key):
        return output
    return test_method
        

result_types = dict(numerical=1,
                    quantity=1 * ureg.s,
                    array=np.array([1, 2, 3]),
                    quantity_array=np.array([1, 2, 3]) * ureg.s,
                    list_of_quantities=[1 * ureg.s, 2 * ureg.s, 3 * ureg.s],
                    two_d_array=np.arange(9).reshape(3, 3),
                    two_d_quantity_array=np.arange(9).reshape(3, 3) * ureg.s,
                    two_d_list_of_quantites=[[i * ureg.s for i in range(3)]
                                             for j in range(3)],
                    tuple_of_quantity_arrays=(np.array([1, 2, 3]) * ureg.s,
                                              np.array([4, 5, 6]) * ureg.m),
                    )
analysis_key_types = dict(numerical=1,
                          quantity=1 * ureg.s,
                          string='test_string',
                          # array=np.array([1, 2, 3]),
                          # quantity_array=np.array([1, 2, 3]) * ureg.s,
                          # list_of_quantities=[1 * ureg.s,
                          #                     2 * ureg.s,
                          #                     3 * ureg.s],
                          )
result_ids = sorted(result_types.keys())
result_ids = sorted(result_types.keys())
key_ids = sorted(analysis_key_types.keys())
test_methods = [make_test_method(result_types[key])
                for key in result_ids]
keys = [analysis_key_types[key] for key in key_ids]
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
            np.testing.assert_array_equal(network.results['test'], result)
        
    @pytest.mark.parametrize('test_method, result', zip(test_methods, results),
                             ids=result_ids)
    def test_returns_existing_result(self, mocker, network, test_method,
                                     result):
        mocker.patch.object(lmt.Network, 'mean_input', new=test_method)
        network.mean_input()
        try:
            assert network.mean_input() == result
        except ValueError:
            np.testing.assert_array_equal(network.mean_input(), result)
            
    def test_result_not_calculated_twice(self, mocker, network):
        mocked = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                              'firing_rates')
        network.firing_rates()
        network.firing_rates()
        mocked.assert_called_once()
                 
    @pytest.mark.parametrize('key', keys, ids=key_ids)
    @pytest.mark.parametrize('result', results, ids=result_ids)
    def test_saves_new_analysis_key_with_param_and_results(self,
                                                           mocker,
                                                           network,
                                                           key,
                                                           result):
        test_method = make_test_method_with_key(result, key)
        mocker.patch.object(lmt.Network, 'mean_input', new=test_method)
        network.mean_input(key)
        try:
            assert network.results['test'][0] == result
        except ValueError:
            np.testing.assert_array_equal(network.results['test'][0], result)
            
    @pytest.mark.parametrize('key', keys, ids=key_ids)
    @pytest.mark.parametrize('result', results, ids=result_ids)
    def test_returns_existing_analysis_key_with_param_and_results(self,
                                                                  mocker,
                                                                  network,
                                                                  key,
                                                                  result):
        test_method = make_test_method_with_key(result, key)
        mocker.patch.object(lmt.Network, 'mean_input', new=test_method)
        network.mean_input(key)
        try:
            assert network.mean_input(key) == result
        except ValueError:
            np.testing.assert_array_equal(network.mean_input(key), result)
    
    @pytest.mark.parametrize('key', keys, ids=key_ids)
    @pytest.mark.parametrize('result', results, ids=result_ids)
    def test_saves_new_param_and_results_for_existing_analysis_key(self,
                                                                   mocker,
                                                                   network,
                                                                   key,
                                                                   result):
        test_method = make_test_method_with_key(result, key)
        mocker.patch.object(lmt.Network, 'mean_input', new=test_method)
        network.mean_input(key)
        network.mean_input(2 * key)
        try:
            assert network.results['test'][1] == result
        except ValueError:
            np.testing.assert_array_equal(network.results['test'][1], result)
        
    def test_returns_existing_key_param_results_for_second_param(self,
                                                                 mocker,
                                                                 network):
        @lmt.Network._check_and_store('test', 'test_key')
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
        @lmt.Network._check_and_store('test', 'test_key')
        def test_method(self, key):
            return key
        mocker.patch('lif_meanfield_tools.Network.mean_input', new=test_method)
        omegas = [10 * ureg.Hz, 11 * ureg.Hz]
        network.mean_input(omegas[0])
        network.mean_input(omegas[1])
        assert len(network.analysis_params['test_key']) == 2
        assert len(network.results['test']) == 2
        
        
meanfield_calls = dict(
    firing_rates=[[], ['firing_rates']],
    mean_input=[[], ['mean']],
    std_input=[[], ['standard_deviation']],
    delay_dist_matrix_multi=[[], ['delay_dist_matrix']],
    delay_dist_matrix_single=[[1 * ureg.Hz], ['delay_dist_matrix']],
    transfer_function_multi=[[], ['transfer_function']],
    transfer_function_single=[[1 * ureg.Hz], ['transfer_function']],
    sensitivity_measure=[[1 * ureg.Hz], ['transfer_function'],
                         'delay_dist_matrix',
                         'sensitivity_measure'],
    power_spectra=[[], ['power_spectra']],
    eigenvalue_spectra=[['MH'], ['eigen_spectra']],
    r_eigenvec_spectra=[['MH'], ['eigen_spectra']],
    l_eigenvec_spectra=[['MH'], ['eigen_spectra']],
    additional_rates_for_fixed_input=[[], [
        'additional_rates_for_fixed_input']],
    fit_transfer_function=[[], ['fit_transfer_function'],
                           'effective_coupling_strength'],
    scan_fit_transfer_function_mean_std_input=[[], [
        'scan_fit_transfer_function_mean_std_input']],
    linear_interpolation_alpha=[[], ['linear_interpolation_alpha']],
    compute_profile_characteristics=[[], ['xi_of_k',
                                          'solve_chareq_rate_boxcar']],
    )

network_calls = dict(
    working_point=['firing_rates', 'mean_input', 'std_input'],
    transfer_function_multi=['mean_input', 'std_input'],
    transfer_function_single=['mean_input', 'std_input'],
    sensitivity_measure=['mean_input', 'std_input'],
    power_spectra=['delay_dist_matrix', 'firing_rates', 'transfer_function'],
    eigenvalue_spectra=['transfer_function', 'delay_dist_matrix'],
    r_eigenvec_spectra=['transfer_function', 'delay_dist_matrix'],
    l_eigenvec_spectra=['transfer_function', 'delay_dist_matrix'],
    fit_transfer_function=['transfer_function', 'mean_input', 'std_input'],
    linear_interpolation_alpha=['mean_input', 'std_input'],
    )

    
methods = sorted(meanfield_calls.keys())
args = [meanfield_calls[method][0] for method in methods]
called_funcs = [meanfield_calls[method][1] for method in methods]


simple_calls = dict(
    firing_rates=['firing_rates'],
    mean_input=['mean'],
    std_input=['standard_deviation'],
    delay_dist_matrix_multi=['delay_dist_matrix'],
    transfer_function_multi=['transfer_function'],
    power_spectra=['power_spectra'],
    compute_profile_characteristics=['xi_of_k', 'solve_chareq_rate_boxcar'],
    )

simple_call_methods = sorted(simple_calls.keys())
simple_call_callees = [simple_calls[method] for method in simple_call_methods]


class Test_functionality:
    
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
        
    def test_compute_profile_characteristics_called_correctly(self, network,
                                                              mocker):
        mock_xi = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                               'xi_of_k')
        mock_xi.return_value = 1, 2, 3, 4
        mock_xi = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                               'solve_chareq_rate_boxcar')
        mock_xi.side_effect = 1, 2
        network.compute_profile_characteristics()
    
    # @pytest.mark.parametrize(
    #     'method, arg, called_funcs', zip(methods, args, called_funcs),
    #     ids=methods)
    # def test_correct_meanfield_functions_are_called(self, mocker, network,
    #                                                 method, arg, called_funcs):
    #     mocked_funcs = []
    #     for func in called_funcs:
    #         mocked_funcs.append(mocker.patch(
    #             'lif_meanfield_tools.meanfield_calcs.{}'.format(func)))
    #     getattr(network, method)(*arg)
    #     for mocked_func in mocked_funcs:
    #         mocked_func.assert_called_once()
    #
    # @pytest.mark.parametrize(
    #     'method, called_funcs', zip(network_calls.keys(),
    #                                 network_calls.values()),
    #     ids=network_calls.keys())
    # def test_correct_network_functions_are_called(self, mocker, network,
    #                                               method, called_funcs):
    #     mocked_funcs = []
    #     for func in called_funcs:
    #         mocked_funcs.append(mocker.patch(
    #             'lif_meanfield_tools.Network.{}'.format(func)))
    #     getattr(network, method)()
    #     for mocked_func in mocked_funcs:
    #         mocked_func.assert_called_once()
