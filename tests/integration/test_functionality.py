import pytest
import numpy as np
from numpy.testing import (
    assert_allclose,
    )

import lif_meanfield_tools as lmt
from ..checks import (assert_quantity_allclose,
                      check_quantity_dicts_are_allclose,
                      check_quantity_dicts_are_equal
                      )


def save_and_load(network, tmpdir):
    temp = tmpdir.mkdir('temp')
    with temp.as_cwd():
        network.save(file='test.h5')
        network.load(file='test.h5')


class Test_Network_functions_give_correct_results:
    
    prefix = 'lif.exp.'
    
    def test_firing_rates_shift(self, network, std_results):
        firing_rates = lmt.lif.exp.firing_rates(network, method='shift')
        assert_allclose(
            firing_rates, std_results[self.prefix + 'firing_rates'])
    
    def test_firing_rates_taylor(self, network, std_results):
        firing_rates = lmt.lif.exp.firing_rates(network, method='taylor')
        assert_allclose(
            firing_rates, std_results[self.prefix + 'firing_rates_taylor'])
        
    def test_mean_input(self, network, std_results):
        lmt.lif.exp.firing_rates(network)
        mean_input = lmt.lif.exp.mean_input(network)
        assert_allclose(mean_input, std_results[self.prefix + 'mean_input'])
        
    def test_standard_deviation(self, network, std_results):
        lmt.lif.exp.firing_rates(network)
        std_input = lmt.lif.exp.std_input(network)
        assert_allclose(std_input, std_results[self.prefix + 'std_input'])
    
    def test_working_point(self, network, std_results):
        working_point = lmt.lif.exp.working_point(network)
        expected_working_point = dict(
            firing_rates=std_results[self.prefix + 'firing_rates'],
            mean_input=std_results[self.prefix + 'mean_input'],
            std_input=std_results[self.prefix + 'std_input'])
        check_quantity_dicts_are_allclose(working_point,
                                          expected_working_point)
                                       
    def test_delay_dist_matrix(self, network, std_results):
        ddm = lmt.models.utils.delay_dist_matrix(network)
        assert_allclose(ddm, std_results['delay_dist_matrix'])
    
    def test_transfer_function_taylor(self, network, std_results):
        lmt.lif.exp.working_point(network)
        transfer_fn = lmt.lif.exp.transfer_function(network, method='taylor')
        assert_allclose(transfer_fn, std_results[self.prefix + 'tf_taylor'])
    
    def test_transfer_function_shift(self, network, std_results):
        lmt.lif.exp.working_point(network)
        transfer_fn = lmt.lif.exp.transfer_function(network, method='shift')
        assert_allclose(transfer_fn, std_results[self.prefix + 'tf_shift'])
    
    def test_transfer_function_single(self, network, std_results):
        lmt.lif.exp.working_point(network)
        transfer_fn = lmt.lif.exp.transfer_function(
            network,
            network.analysis_params['omega'])
        assert_allclose(
            transfer_fn, std_results[self.prefix + 'tf_single'])
    
    def test_sensitivity_measure(self, network, std_results):
        lmt.lif.exp.working_point(network)
        lmt.lif.exp.transfer_function(network)
        lmt.lif.exp.effective_connectivity(network)
        sm = lmt.lif.exp.sensitivity_measure(
            network,)
        assert_allclose(sm, std_results[self.prefix + 'sensitivity_measure'])
    
    def test_power_spectra(self, network, std_results):
        lmt.lif.exp.working_point(network)
        lmt.lif.exp.transfer_function(network)
        lmt.lif.exp.effective_connectivity(network)
        ps = lmt.lif.exp.power_spectra(network)
        assert_allclose(ps, std_results[self.prefix + 'power_spectra'])
    
    @pytest.mark.xfail
    def test_additional_rates_for_fixed_input(self, network, std_results):
        mean_input_set = network.network_params['mean_input_set']
        std_input_set = network.network_params['std_input_set']
        nu_e_ext, nu_i_ext = network.additional_rates_for_fixed_input(
            mean_input_set, std_input_set)
        assert_allclose(nu_e_ext, std_results[self.prefix + 'nu_e_ext'])
        assert_allclose(nu_i_ext, std_results[self.prefix + 'nu_i_ext'])


class Test_saving_and_loading:
    
    prefix = 'lif.exp.'
    
    def test_firing_rates(self, network, tmpdir):
        saved = lmt.lif.exp.firing_rates(network)
        save_and_load(network, tmpdir)
        loaded = network.results[self.prefix + 'firing_rates']
        assert_quantity_allclose(saved, loaded)
        
    def test_mean_input(self, network, tmpdir):
        lmt.lif.exp.firing_rates(network)
        saved = lmt.lif.exp.mean_input(network)
        save_and_load(network, tmpdir)
        loaded = network.results[self.prefix + 'mean_input']
        assert_quantity_allclose(saved, loaded)
        
    def test_standard_deviation(self, network, tmpdir):
        lmt.lif.exp.firing_rates(network)
        saved = lmt.lif.exp.std_input(network)
        save_and_load(network, tmpdir)
        loaded = network.results[self.prefix + 'std_input']
        assert_quantity_allclose(saved, loaded)
    
    def test_working_point(self, network, tmpdir):
        saved = lmt.lif.exp.working_point(network)
        save_and_load(network, tmpdir)
        loaded_fr = network.results[self.prefix + 'firing_rates']
        loaded_mean = network.results[self.prefix + 'mean_input']
        loaded_std = network.results[self.prefix + 'std_input']
        loaded = dict(firing_rates=loaded_fr,
                      mean_input=loaded_mean,
                      std_input=loaded_std)
        check_quantity_dicts_are_equal(saved, loaded)
    
    @pytest.mark.xfail
    def test_delay_dist_matrix(self, network, tmpdir):
        saved = network.delay_dist_matrix()
        save_and_load(network, tmpdir)
        loaded = network.results[self.prefix + 'delay_dist_matrix']
        assert_quantity_allclose(saved, loaded)
    
    @pytest.mark.xfail
    def test_delay_dist_matrix_single(self, network, tmpdir):
        saved = network.delay_dist_matrix(network.analysis_params['omega'])
        save_and_load(network, tmpdir)
        loaded = network.results[self.prefix + 'delay_dist_matrix_single']
        assert_quantity_allclose(saved, loaded)
    
    def test_transfer_function(self, network, tmpdir):
        lmt.lif.exp.working_point(network)
        saved = lmt.lif.exp.transfer_function(network)
        save_and_load(network, tmpdir)
        loaded = network.results[self.prefix + 'transfer_function']
        assert_quantity_allclose(saved, loaded)
    
    @pytest.mark.xfail
    def test_transfer_function_single(self, network, tmpdir):
        network.working_point()
        saved = network.transfer_function(network.analysis_params['omega'])
        save_and_load(network, tmpdir)
        loaded = network.results['transfer_function_single']
        assert_quantity_allclose(saved, loaded)
    
    def test_sensitivity_measure(self, network, tmpdir):
        lmt.lif.exp.working_point(network)
        lmt.lif.exp.transfer_function(network)
        lmt.lif.exp.effective_connectivity(network)
        sm_saved = lmt.lif.exp.sensitivity_measure(network)
        save_and_load(network, tmpdir)
        sm_loaded = network.results[self.prefix + 'sensitivity_measure']
        assert_quantity_allclose(sm_saved, sm_loaded)
    
    def test_power_spectra(self, network, tmpdir):
        lmt.lif.exp.working_point(network)
        lmt.lif.exp.transfer_function(network)
        lmt.lif.exp.effective_connectivity(network)
        saved = lmt.lif.exp.power_spectra(network)
        save_and_load(network, tmpdir)
        loaded = network.results[self.prefix + 'power_spectra']
        assert_quantity_allclose(saved, loaded)
    
    @pytest.mark.xfail
    def test_eigenvalue_spectra(self, network, tmpdir):
        network.working_point()
        network.delay_dist_matrix()
        network.transfer_function()
        saved = network.eigenvalue_spectra('MH')
        save_and_load(network, tmpdir)
        loaded = network.results['eigenvalue_spectra']
        assert_quantity_allclose(saved, loaded)
    
    @pytest.mark.xfail
    def test_r_eigenvec_spectra(self, network, tmpdir):
        network.working_point()
        network.delay_dist_matrix()
        network.transfer_function()
        saved = network.r_eigenvec_spectra('MH')
        save_and_load(network, tmpdir)
        loaded = network.results['r_eigenvec_spectra']
        assert_quantity_allclose(saved, loaded)
    
    @pytest.mark.xfail
    def test_l_eigenvec_spectra(self, network, tmpdir):
        network.working_point()
        network.delay_dist_matrix()
        network.transfer_function()
        saved = network.l_eigenvec_spectra('MH')
        save_and_load(network, tmpdir)
        loaded = network.results['l_eigenvec_spectra']
        assert_quantity_allclose(saved, loaded)
    
    @pytest.mark.xfail
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

    @pytest.mark.xfail
    def test_results_hash_dict_same_before_and_after_saving_and_loading(
            self, network, tmpdir):
        network.firing_rates()
        rhd = network.results_hash_dict
        save_and_load(network, tmpdir)
        check_quantity_dicts_are_equal(rhd, network.results_hash_dict)
        
    @pytest.mark.xfail
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
    
    prefix = 'lif.exp.'
    
    def test_firing_rates(self, network):
        firing_rates = lmt.lif.exp.firing_rates(network)
        assert_allclose(firing_rates,
                        network.results[self.prefix + 'firing_rates'])
    
    def test_mean_input(self, network):
        lmt.lif.exp.firing_rates(network)
        mean_input = lmt.lif.exp.mean_input(network)
        assert_allclose(mean_input,
                        network.results[self.prefix + 'mean_input'])
    
    def test_std_input(self, network):
        lmt.lif.exp.firing_rates(network)
        std_input = lmt.lif.exp.std_input(network)
        assert_allclose(std_input, network.results[self.prefix + 'std_input'])
    
    def test_working_point(self, network):
        working_point = lmt.lif.exp.working_point(network)
        assert_allclose(working_point['firing_rates'],
                        network.results[self.prefix + 'firing_rates'])
        assert_allclose(working_point['mean_input'],
                        network.results[self.prefix + 'mean_input'])
        assert_allclose(working_point['std_input'],
                        network.results[self.prefix + 'std_input'])
    
    def test_delay_dist_matrix(self, network):
        delay_dist_matrix = lmt.models.utils.delay_dist_matrix(network)
        assert_allclose(delay_dist_matrix,
                        network.network_params['D'])

    def test_transfer_function(self, network):
        lmt.lif.exp.working_point(network)
        transfer_function_taylor = lmt.lif.exp.transfer_function(
            network, method='taylor')
        assert_allclose(transfer_function_taylor,
                        network.results[self.prefix + 'transfer_function'])
        transfer_function_shift = lmt.lif.exp.transfer_function(
            network, method='shift')
        assert_allclose(transfer_function_shift,
                        network.results[self.prefix + 'transfer_function'])
    
    def test_transfer_function_single(self, network):
        lmt.lif.exp.working_point(network)
        omega = network.analysis_params['omega']
        transfer_function_taylor = lmt.lif.exp.transfer_function(
            network, omega, method='taylor')
        assert_allclose(
            transfer_function_taylor,
            network.results[self.prefix + 'transfer_function']
            )
        transfer_function_shift = lmt.lif.exp.transfer_function(
            network, omega, method='shift')
        assert_allclose(
            transfer_function_shift,
            network.results[self.prefix + 'transfer_function']
            )
    
    def test_sensitivity_measure(self, network):
        lmt.lif.exp.working_point(network)
        lmt.lif.exp.transfer_function(network)
        lmt.lif.exp.effective_connectivity(network)
        sensitivity_measure = lmt.lif.exp.sensitivity_measure(network)
        assert_allclose(sensitivity_measure,
                        network.results[self.prefix + 'sensitivity_measure'])
    
    def test_power_spectra(self, network):
        lmt.lif.exp.working_point(network)
        lmt.lif.exp.transfer_function(network)
        lmt.lif.exp.effective_connectivity(network)
        power_spectra = lmt.lif.exp.power_spectra(network)
        assert_allclose(power_spectra,
                        network.results[self.prefix + 'power_spectra'])

    @pytest.mark.xfail
    def test_additional_rates_for_fixed_input(self, network):
        mean_input_set = network.network_params['mean_input_set']
        std_input_set = network.network_params['std_input_set']
        nu_e_ext, nu_i_ext = network.additional_rates_for_fixed_input(
            mean_input_set, std_input_set)
        assert_allclose(nu_e_ext, network.results[self.prefix + 'nu_e_ext'])
        assert_allclose(nu_i_ext, network.results[self.prefix + 'nu_i_ext'])


class Test_correct_return_value_for_second_call:
    
    @pytest.mark.parametrize('method', ['taylor', 'shift'])
    def test_transfer_function(self, network, method):
        lmt.lif.exp.working_point(network)
        transfer_function_taylor_0 = lmt.lif.exp.transfer_function(
            network, method=method)
        transfer_function_taylor_1 = lmt.lif.exp.transfer_function(
            network, method=method)
        assert_quantity_allclose(transfer_function_taylor_0,
                                 transfer_function_taylor_1)

    def test_transfer_function_first_taylor_then_shift(self, network):
        lmt.lif.exp.working_point(network)
        transfer_function_taylor = lmt.lif.exp.transfer_function(
            network, method='taylor')
        transfer_function_shift = lmt.lif.exp.transfer_function(
            network, method='shift')
        with pytest.raises(AssertionError):
            assert_quantity_allclose(transfer_function_taylor,
                                     transfer_function_shift)

    def test_transfer_function_single_first_taylor_then_shift(self, network):
        omega = network.analysis_params['omega']
        lmt.lif.exp.working_point(network)
        tf_taylor = lmt.lif.exp.transfer_function(
            network, omega, method='taylor')
        tf_shift = lmt.lif.exp.transfer_function(
            network, omega, method='shift')
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
        network = lmt.models.Microcircuit(negative_rate_params_file,
                                            analysis_params_file)
        firing_rates = lmt.lif.exp.firing_rates(network)
        assert not any(firing_rates < 0)
