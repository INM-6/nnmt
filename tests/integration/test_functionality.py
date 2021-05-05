import pytest

import lif_meanfield_tools as lmt
from ..checks import (assert_quantity_array_equal,
                      assert_quantity_allclose,
                      check_quantity_dicts_are_allclose,
                      check_quantity_dicts_are_equal
                      )


def save_and_load(network, tmpdir):
    temp = tmpdir.mkdir('temp')
    with temp.as_cwd():
        network.save(file='test.h5')
        network.load(file='test.h5')


class Test_Network_functions_give_correct_results:
    
    def test_firing_rates_shift(self, network, std_results):
        firing_rates = network.firing_rates(method='shift')
        assert_quantity_allclose(
            firing_rates, std_results['firing_rates'])
    
    def test_firing_rates_taylor(self, network, std_results):
        firing_rates = network.firing_rates(method='taylor')
        assert_quantity_allclose(
            firing_rates, std_results['firing_rates_taylor'])
        
    def test_mean_input(self, network, std_results):
        network.firing_rates()
        mean_input = network.mean_input()
        assert_quantity_allclose(mean_input,
                                    std_results['mean_input'])
        
    def test_standard_deviation(self, network, std_results):
        network.firing_rates()
        std_input = network.std_input()
        assert_quantity_allclose(std_input, std_results['std_input'])
    
    def test_working_point(self, network, std_results):
        working_point = network.working_point()
        expected_working_point = dict(firing_rates=std_results['firing_rates'],
                                      mean_input=std_results['mean_input'],
                                      std_input=std_results['std_input'])
        check_quantity_dicts_are_allclose(working_point,
                                       expected_working_point)
                                       
    def test_delay_dist_matrix(self, network, std_results):
        ddm = network.delay_dist_matrix()
        assert_quantity_allclose(ddm, std_results['delay_dist_matrix'])
    
    def test_delay_dist_matrix_single(self, network, std_results):
        ddm = network.delay_dist_matrix(network.analysis_params['omega'])
        assert_quantity_allclose(ddm,
                                    std_results['delay_dist_matrix_single'])
    
    def test_transfer_function_taylor(self, network, std_results):
        network.working_point()
        transfer_fn = network.transfer_function(method='taylor')
        assert_quantity_allclose(transfer_fn, std_results['tf_taylor'])
    
    def test_transfer_function_shift(self, network, std_results):
        network.working_point()
        transfer_fn = network.transfer_function(method='shift')
        assert_quantity_allclose(transfer_fn, std_results['tf_shift'])
    
    def test_transfer_function_single(self, network, std_results):
        network.working_point()
        transfer_fn = network.transfer_function(
            network.analysis_params['omega'])
        assert_quantity_allclose(
            transfer_fn, std_results['transfer_function_single'])
    
    def test_sensitivity_measure(self, network, std_results):
        network.working_point()
        sm = network.sensitivity_measure(network.analysis_params['omega'])
        assert_quantity_allclose(sm, std_results['sensitivity_measure'])
    
    def test_power_spectra(self, network, std_results):
        network.working_point()
        network.transfer_function()
        network.delay_dist_matrix()
        ps = network.power_spectra()
        assert_quantity_allclose(ps, std_results['power_spectra'])
    
    def test_eigenvalue_spectra(self, network, std_results):
        network.working_point()
        network.delay_dist_matrix()
        network.transfer_function()
        es = network.eigenvalue_spectra('MH')
        assert_quantity_allclose(
            es, std_results['eigenvalue_spectra_MH'])
    
    def test_r_eigenvec_spectra(self, network, std_results):
        network.working_point()
        network.delay_dist_matrix()
        network.transfer_function()
        es = network.r_eigenvec_spectra('MH')
        assert_quantity_allclose(
            es, std_results['r_eigenvec_spectra_MH'])
    
    def test_l_eigenvec_spectra(self, network, std_results):
        network.working_point()
        network.delay_dist_matrix()
        network.transfer_function(method='shift')
        es = network.l_eigenvec_spectra('MH')
        assert_quantity_allclose(
            es, std_results['l_eigenvec_spectra_MH'])
    
    def test_additional_rates_for_fixed_input(self, network, std_results):
        mean_input_set = network.network_params['mean_input_set']
        std_input_set = network.network_params['std_input_set']
        nu_e_ext, nu_i_ext = network.additional_rates_for_fixed_input(
            mean_input_set, std_input_set)
        assert_quantity_allclose(nu_e_ext, std_results['nu_e_ext'])
        assert_quantity_allclose(nu_i_ext, std_results['nu_i_ext'])


class Test_saving_and_loading:
    
    def test_firing_rates(self, network, tmpdir):
        saved = network.firing_rates()
        save_and_load(network, tmpdir)
        loaded = network.results['firing_rates']
        assert_quantity_allclose(saved, loaded)
        
    def test_mean_input(self, network, tmpdir):
        network.firing_rates()
        saved = network.mean_input()
        save_and_load(network, tmpdir)
        loaded = network.results['mean_input']
        assert_quantity_allclose(saved, loaded)
        
    def test_standard_deviation(self, network, tmpdir):
        network.firing_rates()
        saved = network.std_input()
        save_and_load(network, tmpdir)
        loaded = network.results['std_input']
        assert_quantity_allclose(saved, loaded)
    
    def test_working_point(self, network, tmpdir):
        saved = network.working_point()
        save_and_load(network, tmpdir)
        loaded_fr = network.results['firing_rates']
        loaded_mean = network.results['mean_input']
        loaded_std = network.results['std_input']
        loaded = dict(firing_rates=loaded_fr,
                      mean_input=loaded_mean,
                      std_input=loaded_std)
        check_quantity_dicts_are_equal(saved, loaded)
    
    def test_delay_dist_matrix(self, network, tmpdir):
        saved = network.delay_dist_matrix()
        save_and_load(network, tmpdir)
        loaded = network.results['delay_dist_matrix']
        assert_quantity_allclose(saved, loaded)
    
    def test_delay_dist_matrix_single(self, network, tmpdir):
        saved = network.delay_dist_matrix(network.analysis_params['omega'])
        save_and_load(network, tmpdir)
        loaded = network.results['delay_dist_matrix_single']
        assert_quantity_allclose(saved, loaded)
    
    def test_transfer_function(self, network, tmpdir):
        network.working_point()
        saved = network.transfer_function()
        save_and_load(network, tmpdir)
        loaded = network.results['transfer_function']
        assert_quantity_allclose(saved, loaded)
    
    def test_transfer_function_single(self, network, tmpdir):
        network.working_point()
        saved = network.transfer_function(network.analysis_params['omega'])
        save_and_load(network, tmpdir)
        loaded = network.results['transfer_function_single']
        assert_quantity_allclose(saved, loaded)
    
    def test_sensitivity_measure(self, network, tmpdir):
        network.working_point()
        sm_saved = network.sensitivity_measure(
            network.analysis_params['omega'])
        save_and_load(network, tmpdir)
        sm_loaded = network.results['sensitivity_measure']
        assert_quantity_allclose(sm_saved, sm_loaded)
    
    def test_power_spectra(self, network, tmpdir):
        network.working_point()
        network.delay_dist_matrix()
        network.transfer_function()
        saved = network.power_spectra()
        save_and_load(network, tmpdir)
        loaded = network.results['power_spectra']
        assert_quantity_allclose(saved, loaded)
    
    def test_eigenvalue_spectra(self, network, tmpdir):
        network.working_point()
        network.delay_dist_matrix()
        network.transfer_function()
        saved = network.eigenvalue_spectra('MH')
        save_and_load(network, tmpdir)
        loaded = network.results['eigenvalue_spectra']
        assert_quantity_allclose(saved, loaded)
    
    def test_r_eigenvec_spectra(self, network, tmpdir):
        network.working_point()
        network.delay_dist_matrix()
        network.transfer_function()
        saved = network.r_eigenvec_spectra('MH')
        save_and_load(network, tmpdir)
        loaded = network.results['r_eigenvec_spectra']
        assert_quantity_allclose(saved, loaded)
    
    def test_l_eigenvec_spectra(self, network, tmpdir):
        network.working_point()
        network.delay_dist_matrix()
        network.transfer_function()
        saved = network.l_eigenvec_spectra('MH')
        save_and_load(network, tmpdir)
        loaded = network.results['l_eigenvec_spectra']
        assert_quantity_allclose(saved, loaded)
    
    def test_additional_rates_for_fixed_input(self, network, tmpdir):
        mean_input_set = network.network_params['mean_input_set']
        std_input_set = network.network_params['std_input_set']
        saved_e, saved_i = network.additional_rates_for_fixed_input(
            mean_input_set, std_input_set)
        save_and_load(network, tmpdir)
        loaded_e = network.results['nu_e_ext']
        loaded_i = network.results['nu_i_ext']
        assert_quantity_allclose(saved_e, loaded_e)
        assert_quantity_allclose(saved_i, loaded_i)

    def test_results_hash_dict_same_before_and_after_saving_and_loading(
            self, network, tmpdir):
        network.firing_rates()
        rhd = network.results_hash_dict
        save_and_load(network, tmpdir)
        check_quantity_dicts_are_equal(rhd, network.results_hash_dict)
        
    def test_results_hash_dict_same_for_new_loading_network(
            self, network, tmpdir):
        network.firing_rates()
        network.results_hash_dict
        with tmpdir.as_cwd():
            network.save('test.h5')
            new_network = lmt.Network(file='test.h5')
        check_quantity_dicts_are_equal(network.results_hash_dict,
                                       new_network.results_hash_dict)


class Test_temporary_storage_of_results:
    
    def test_firing_rates(self, network):
        firing_rates = network.firing_rates()
        assert_quantity_allclose(firing_rates,
                                    network.results['firing_rates'])
    
    def test_mean_input(self, network):
        network.firing_rates()
        mean_input = network.mean_input()
        assert_quantity_allclose(mean_input,
                                    network.results['mean_input'])
    
    def test_std_input(self, network):
        network.firing_rates()
        std_input = network.std_input()
        assert_quantity_allclose(std_input,
                                    network.results['std_input'])
    
    def test_working_point(self, network):
        working_point = network.working_point()
        assert_quantity_allclose(working_point['firing_rates'],
                                    network.results['firing_rates'])
        assert_quantity_allclose(working_point['mean_input'],
                                    network.results['mean_input'])
        assert_quantity_allclose(working_point['std_input'],
                                    network.results['std_input'])
    
    def test_delay_dist_matrix(self, network):
        delay_dist_matrix = network.delay_dist_matrix()
        assert_quantity_allclose(delay_dist_matrix,
                                    network.results['delay_dist_matrix'])
    
    def test_delay_dist_matrix_single(self, network):
        omega = network.analysis_params['omega']
        delay_dist_matrix = network.delay_dist_matrix(omega)
        assert_quantity_allclose(
            delay_dist_matrix, network.results['delay_dist_matrix_single'])
    
    def test_transfer_function(self, network):
        network.working_point()
        transfer_function_taylor = network.transfer_function(method='taylor')
        assert_quantity_allclose(transfer_function_taylor,
                                    network.results['transfer_function'])
        transfer_function_shift = network.transfer_function(method='shift')
        assert_quantity_allclose(transfer_function_shift,
                                    network.results['transfer_function'])
    
    def test_transfer_function_single(self, network):
        network.working_point()
        omega = network.analysis_params['omega']
        transfer_function_taylor = network.transfer_function(omega,
                                                             method='taylor')
        assert_quantity_allclose(
            transfer_function_taylor,
            network.results['transfer_function_single']
            )
        transfer_function_shift = network.transfer_function(omega,
                                                            method='shift')
        assert_quantity_allclose(
            transfer_function_shift,
            network.results['transfer_function_single']
            )
    
    def test_sensitivity_measure(self, network):
        network.working_point()
        omega = network.analysis_params['omega']
        sensitivity_measure = network.sensitivity_measure(omega)
        assert_quantity_allclose(sensitivity_measure,
                                    network.results['sensitivity_measure'])
    
    def test_power_spectra(self, network):
        network.working_point()
        network.delay_dist_matrix()
        network.transfer_function()
        power_spectra = network.power_spectra()
        assert_quantity_allclose(power_spectra,
                                    network.results['power_spectra'])
    
    def test_eigenvalue_spectra(self, network):
        network.working_point()
        network.delay_dist_matrix()
        network.transfer_function()
        eigenvalue_spectra_mh = network.eigenvalue_spectra('MH')
        assert_quantity_allclose(eigenvalue_spectra_mh,
                                    network.results['eigenvalue_spectra'])
        eigenvalue_spectra_prop = network.eigenvalue_spectra('prop')
        assert_quantity_allclose(eigenvalue_spectra_prop,
                                    network.results['eigenvalue_spectra'])
        eigenvalue_spectra_prop_inv = network.eigenvalue_spectra('prop_inv')
        assert_quantity_allclose(
            eigenvalue_spectra_prop_inv,
            network.results['eigenvalue_spectra'])
        
    def test_r_eigenvec_spectra(self, network):
        network.working_point()
        network.delay_dist_matrix()
        network.transfer_function()
        r_eigenvec_spectra_mh = network.r_eigenvec_spectra('MH')
        assert_quantity_allclose(r_eigenvec_spectra_mh,
                                    network.results['r_eigenvec_spectra'])
        r_eigenvec_spectra_prop = network.r_eigenvec_spectra('prop')
        assert_quantity_allclose(r_eigenvec_spectra_prop,
                                    network.results['r_eigenvec_spectra'])
        r_eigenvec_spectra_prop_inv = network.r_eigenvec_spectra('prop_inv')
        assert_quantity_allclose(r_eigenvec_spectra_prop_inv,
                                    network.results['r_eigenvec_spectra'])
        
    def test_l_eigenvec_spectra(self, network):
        network.working_point()
        network.delay_dist_matrix()
        network.transfer_function()
        l_eigenvec_spectra_mh = network.l_eigenvec_spectra('MH')
        assert_quantity_allclose(l_eigenvec_spectra_mh,
                                    network.results['l_eigenvec_spectra'])
        l_eigenvec_spectra_prop = network.l_eigenvec_spectra('prop')
        assert_quantity_allclose(l_eigenvec_spectra_prop,
                                    network.results['l_eigenvec_spectra'])
        l_eigenvec_spectra_prop_inv = network.l_eigenvec_spectra('prop_inv')
        assert_quantity_allclose(l_eigenvec_spectra_prop_inv,
                                    network.results['l_eigenvec_spectra'])

    def test_additional_rates_for_fixed_input(self, network):
        mean_input_set = network.network_params['mean_input_set']
        std_input_set = network.network_params['std_input_set']
        nu_e_ext, nu_i_ext = network.additional_rates_for_fixed_input(
            mean_input_set, std_input_set)
        assert_quantity_allclose(nu_e_ext, network.results['nu_e_ext'])
        assert_quantity_allclose(nu_i_ext, network.results['nu_i_ext'])


class Test_correct_return_value_for_second_call:
    
    @pytest.mark.parametrize('method', ['taylor', 'shift'])
    def test_transfer_function(self, network, method):
        network.working_point()
        transfer_function_taylor_0 = network.transfer_function(method=method)
        transfer_function_taylor_1 = network.transfer_function(method=method)
        assert_quantity_allclose(transfer_function_taylor_0,
                                    transfer_function_taylor_1)

    def test_transfer_function_first_taylor_then_shift(self, network):
        network.working_point()
        transfer_function_taylor = network.transfer_function(method='taylor')
        transfer_function_shift = network.transfer_function(method='shift')
        with pytest.raises(AssertionError):
            assert_quantity_allclose(transfer_function_taylor,
                                        transfer_function_shift)

    def test_transfer_function_single_first_taylor_then_shift(self, network):
        omega = network.analysis_params['omega']
        network.working_point()
        tf_taylor = network.transfer_function(omega, method='taylor')
        tf_shift = network.transfer_function(omega, method='shift')
        with pytest.raises(AssertionError):
            assert_quantity_allclose(tf_taylor, tf_shift)
            
            
class Test_negative_firing_rate_regime:
    """
    These tests where implemented, because we encountered the situation where
    the firing rate function returned negative results, which does not make
    sense. Therefore we here check the firing rate for the parameters for which
    these false results occurred.
    """
    
    def test_no_negative_firing_rates(self):
        negative_rate_params_file = ('tests/fixtures/integration/config/'
                                     'minimal_negative.yaml')
        analysis_params_file = ('tests/fixtures/integration/config/'
                                'analysis_params.yaml')
        network = lmt.Network(negative_rate_params_file,
                              analysis_params_file)
        firing_rates = network.firing_rates()
        assert not any(firing_rates < 0)
