import pytest
import numpy as np
import copy
from numpy.testing import (
    assert_allclose
    )

import nnmt
from ..checks import (assert_quantity_allclose,
                      check_quantity_dicts_are_allclose,
                      check_quantity_dicts_are_equal
                      )


def save_and_load(network, tmpdir):
    temp = tmpdir.mkdir('temp')
    with temp.as_cwd():
        network.save(file='test.h5')
        network.load(file='test.h5')


class Test_lif_exp_functions_give_correct_results:

    prefix = 'lif.exp.'

    def test_firing_rates_shift_ODE(self, network, std_results):
        firing_rates = nnmt.lif.exp.firing_rates(network, method='shift',
                                                 fixpoint_method='ODE')
        print(firing_rates)
        assert_allclose(
            firing_rates, std_results[self.prefix + 'firing_rates'])

    def test_firing_rates_shift_LSTSQ(self, network, std_results):
        nu_0 = [1.0, 3.0, 4.0, 5.0, 7.0, 8.0, 1.0, 7.0]
        firing_rates = nnmt.lif.exp.firing_rates(network, method='shift',
                                                 fixpoint_method='LSTSQ',
                                                 nu_0=nu_0)
        assert_allclose(
            firing_rates, std_results[self.prefix + 'firing_rates'])

    def test_firing_rates_taylor(self, network, std_results):
        firing_rates = nnmt.lif.exp.firing_rates(network, method='taylor')
        assert_allclose(
            firing_rates, std_results[self.prefix + 'firing_rates_taylor'])

    def test_firing_rates_fully_vectorized(self):
        network = nnmt.models.Network(
            file=('tests/fixtures/integration/data/lif_exp/'
                  'firing_rates_fully_vectorized.h5'))
        old_results = network.results['lif.exp.firing_rates']
        network.clear_results()
        new_results = nnmt.lif.exp.firing_rates(network)
        assert_allclose(
            old_results, new_results)

    def test_firing_rates_with_external_dc_current(self):
        network = nnmt.models.Network(
            file=('tests/fixtures/integration/data/lif_exp/'
                  'firing_rates_with_external_dc_current.h5'))
        old_results = network.results['lif.exp.firing_rates']
        network.clear_results()
        new_results = nnmt.lif.exp.firing_rates(network)
        assert_allclose(
            old_results, new_results)

    def test_firing_rates_with_zero_external_dc_current(self):
        network = nnmt.models.Network(
            file=('tests/fixtures/integration/data/lif_exp/'
                  'firing_rates_fully_vectorized.h5'))
        old_results = network.results['lif.exp.firing_rates']
        network.clear_results()
        network.network_params['C'] = 1e-12
        network.network_params['I_ext'] = 0e-12
        new_results = nnmt.lif.exp.firing_rates(network)
        assert_allclose(
            old_results, new_results)

    def test_mean_input(self, network, std_results):
        nnmt.lif.exp.firing_rates(network)
        mean_input = nnmt.lif.exp.mean_input(network)
        assert_allclose(mean_input, std_results[self.prefix + 'mean_input'])

    def test_standard_deviation(self, network, std_results):
        nnmt.lif.exp.firing_rates(network)
        std_input = nnmt.lif.exp.std_input(network)
        assert_allclose(std_input, std_results[self.prefix + 'std_input'])

    def test_working_point(self, network, std_results):
        working_point = nnmt.lif.exp.working_point(network)
        expected_working_point = dict(
            firing_rates=std_results[self.prefix + 'firing_rates'],
            mean_input=std_results[self.prefix + 'mean_input'],
            std_input=std_results[self.prefix + 'std_input'])
        check_quantity_dicts_are_allclose(working_point,
                                          expected_working_point)

    def test_transfer_function_taylor(self, network, std_results):
        nnmt.lif.exp.working_point(network)
        transfer_fn = nnmt.lif.exp.transfer_function(network, method='taylor')
        assert_allclose(transfer_fn, std_results[self.prefix + 'tf_taylor'])

    def test_transfer_function_shift(self, network, std_results):
        nnmt.lif.exp.working_point(network)
        transfer_fn = nnmt.lif.exp.transfer_function(network, method='shift')
        assert_allclose(transfer_fn, std_results[self.prefix + 'tf_shift'])

    def test_transfer_function_single(self, network, std_results):
        nnmt.lif.exp.working_point(network)
        transfer_fn = nnmt.lif.exp.transfer_function(
            network,
            network.analysis_params['omega'])
        assert_allclose(
            transfer_fn, std_results[self.prefix + 'tf_single'])

    def test_sensitivity_measure(self, network, std_results):
        nnmt.lif.exp.working_point(network)
        nnmt.lif.exp.transfer_function(network)
        nnmt.network_properties.delay_dist_matrix(network)
        nnmt.lif.exp.effective_connectivity(network)
        frequency = network.analysis_params['omega']/(2*np.pi)
        sm = nnmt.lif.exp.sensitivity_measure(
            network, frequency=frequency)

        sm_fix = std_results[self.prefix + 'sensitivity_measure']
        check_quantity_dicts_are_allclose(sm, sm_fix)

    def test_sensitivity_measure_all_eigenmodes(self, network, std_results):
        nnmt.lif.exp.working_point(network)
        nnmt.lif.exp.transfer_function(network)
        nnmt.network_properties.delay_dist_matrix(network)
        nnmt.lif.exp.effective_connectivity(network)
        margin = network.analysis_params['margin']
        sm = nnmt.lif.exp.sensitivity_measure_all_eigenmodes(
            network, margin=margin)

        sm_fix = std_results[self.prefix + 'sensitivity_measure_all_eigenmodes']
        check_quantity_dicts_are_allclose(sm, sm_fix)

    def test_power_spectra(self, network, std_results):
        nnmt.lif.exp.working_point(network)
        nnmt.lif.exp.transfer_function(network)
        nnmt.network_properties.delay_dist_matrix(network)
        nnmt.lif.exp.effective_connectivity(network)
        ps = nnmt.lif.exp.power_spectra(network)
        assert_allclose(ps, std_results[self.prefix + 'power_spectra'])

    @pytest.mark.xfail
    def test_additional_rates_for_fixed_input(self, network, std_results):
        mean_input_set = network.network_params['mean_input_set']
        std_input_set = network.network_params['std_input_set']
        nu_e_ext, nu_i_ext = network.additional_rates_for_fixed_input(
            mean_input_set, std_input_set)
        assert_allclose(nu_e_ext, std_results[self.prefix + 'nu_e_ext'])
        assert_allclose(nu_i_ext, std_results[self.prefix + 'nu_i_ext'])

    def test_cvs(self, network, std_results):
        nnmt.lif.exp.working_point(network)
        cvs = nnmt.lif.exp.cvs(network)
        assert_allclose(cvs, std_results[self.prefix + 'cvs'])

    def test_pairwise_effective_connectivity_and_spectral_bound_and_pairwise_covariances(self):

        network = nnmt.models.Plain(
            file=('tests/fixtures/integration/data/lif_exp/'
                  'spectral_bound_and_pairwise_covariances.h5'))
        W_old = network.results['lif.exp.pairwise_effective_connectivity']
        r_old = network.results['lif.exp.spectral_bound']
        covs_old = network.results['lif.exp.pairwise_covariances']
        network.clear_results()
        nnmt.lif.exp.working_point(network)
        nnmt.lif.exp.cvs(network)
        W = nnmt.lif.exp.pairwise_effective_connectivity(network)
        assert_allclose(W_old, W)
        r = nnmt.lif.exp.spectral_bound(network)
        assert_allclose(r_old, r)
        covs = nnmt.lif.exp.pairwise_covariances(network)
        assert_allclose(covs_old, covs)


class Test_network_properties:

    @pytest.mark.parametrize(
        'fixtures',
        ['tests/fixtures/integration/data/network_properties/delay_none.h5',
         'tests/fixtures/integration/data/network_properties/delay_truncated_gaussian.h5',
         'tests/fixtures/integration/data/network_properties/delay_gaussian.h5',
         'tests/fixtures/integration/data/network_properties/delay_lognormal.h5',
         ])
    def test_delay_dist_matrix(self, fixtures):
        # ddm = nnmt.network_properties.delay_dist_matrix(network)
        # assert_allclose(ddm, std_results['delay_dist_matrix'])
        # load old results
        network = nnmt.models.Network(file=fixtures)
        old_results = network.results['D']
        # create new empty network with same params and calc results
        network_params = copy.deepcopy(network.network_params)
        analysis_params = copy.deepcopy(network.analysis_params)
        new_network = nnmt.models.Network(network_params, analysis_params)
        new_results = nnmt.network_properties.delay_dist_matrix(new_network)

        assert_allclose(
            old_results, new_results)


class Test_saving_and_loading:

    prefix = 'lif.exp.'

    def test_firing_rates(self, network, tmpdir):
        saved = nnmt.lif.exp.firing_rates(network)
        save_and_load(network, tmpdir)
        loaded = network.results[self.prefix + 'firing_rates']
        assert_quantity_allclose(saved, loaded)

    def test_mean_input(self, network, tmpdir):
        nnmt.lif.exp.firing_rates(network)
        saved = nnmt.lif.exp.mean_input(network)
        save_and_load(network, tmpdir)
        loaded = network.results[self.prefix + 'mean_input']
        assert_quantity_allclose(saved, loaded)

    def test_standard_deviation(self, network, tmpdir):
        nnmt.lif.exp.firing_rates(network)
        saved = nnmt.lif.exp.std_input(network)
        save_and_load(network, tmpdir)
        loaded = network.results[self.prefix + 'std_input']
        assert_quantity_allclose(saved, loaded)

    def test_working_point(self, network, tmpdir):
        saved = nnmt.lif.exp.working_point(network)
        save_and_load(network, tmpdir)
        loaded_fr = network.results[self.prefix + 'firing_rates']
        loaded_mean = network.results[self.prefix + 'mean_input']
        loaded_std = network.results[self.prefix + 'std_input']
        loaded = dict(firing_rates=loaded_fr,
                      mean_input=loaded_mean,
                      std_input=loaded_std)
        check_quantity_dicts_are_equal(saved, loaded)

    def test_delay_dist_matrix(self, network, tmpdir):
        saved = nnmt.network_properties.delay_dist_matrix(network)
        save_and_load(network, tmpdir)
        loaded = network.results['D']
        assert_quantity_allclose(saved, loaded)

    def test_delay_dist_matrix_single(self, network, tmpdir):
        saved = nnmt.network_properties.delay_dist_matrix(
            network, [network.analysis_params['omega'] / 2 / np.pi])
        save_and_load(network, tmpdir)
        loaded = network.results['D']
        assert_quantity_allclose(saved, loaded)

    def test_transfer_function(self, network, tmpdir):
        nnmt.lif.exp.working_point(network)
        saved = nnmt.lif.exp.transfer_function(network)
        save_and_load(network, tmpdir)
        loaded = network.results[self.prefix + 'transfer_function']
        assert_quantity_allclose(saved, loaded)

    def test_transfer_function_single(self, network, tmpdir):
        nnmt.lif.exp.working_point(network)
        saved = nnmt.lif.exp.transfer_function(
            network, network.analysis_params['omega'] / 2 / np.pi)
        save_and_load(network, tmpdir)
        loaded = network.results[self.prefix + 'transfer_function']
        assert_quantity_allclose(saved, loaded)

    def test_sensitivity_measure(self, network, tmpdir):
        nnmt.lif.exp.working_point(network)
        nnmt.lif.exp.transfer_function(network)
        nnmt.network_properties.delay_dist_matrix(network)
        nnmt.lif.exp.effective_connectivity(network)
        frequency = network.analysis_params['omega']/(2*np.pi)
        sm_saved = nnmt.lif.exp.sensitivity_measure(network,
                                                    frequency=frequency)
        save_and_load(network, tmpdir)
        sm_loaded = network.results[self.prefix + 'sensitivity_measure']
        check_quantity_dicts_are_allclose(sm_saved, sm_loaded)

    def test_sensitivity_measure_all_eigenmodes(self, network, tmpdir):
        nnmt.lif.exp.working_point(network)
        nnmt.lif.exp.transfer_function(network)
        nnmt.network_properties.delay_dist_matrix(network)
        nnmt.lif.exp.effective_connectivity(network)
        margin = network.analysis_params['margin']
        sm_saved = nnmt.lif.exp.sensitivity_measure_all_eigenmodes(
            network, margin=margin)

        save_and_load(network, tmpdir)
        sm_loaded = network.results[
            self.prefix + 'sensitivity_measure_all_eigenmodes']
        check_quantity_dicts_are_allclose(sm_saved, sm_loaded)

    def test_power_spectra(self, network, tmpdir):
        nnmt.lif.exp.working_point(network)
        nnmt.lif.exp.transfer_function(network)
        nnmt.network_properties.delay_dist_matrix(network)
        nnmt.lif.exp.effective_connectivity(network)
        saved = nnmt.lif.exp.power_spectra(network)
        save_and_load(network, tmpdir)
        loaded = network.results[self.prefix + 'power_spectra']
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
            new_network = nnmt.Network(file='test.h5')
        check_quantity_dicts_are_equal(network.results_hash_dict,
                                       new_network.results_hash_dict)


class Test_temporary_storage_of_results:

    prefix = 'lif.exp.'

    def test_firing_rates(self, network):
        firing_rates = nnmt.lif.exp.firing_rates(network)
        assert_allclose(firing_rates,
                        network.results[self.prefix + 'firing_rates'])

    def test_mean_input(self, network):
        nnmt.lif.exp.firing_rates(network)
        mean_input = nnmt.lif.exp.mean_input(network)
        assert_allclose(mean_input,
                        network.results[self.prefix + 'mean_input'])

    def test_std_input(self, network):
        nnmt.lif.exp.firing_rates(network)
        std_input = nnmt.lif.exp.std_input(network)
        assert_allclose(std_input, network.results[self.prefix + 'std_input'])

    def test_working_point(self, network):
        working_point = nnmt.lif.exp.working_point(network)
        assert_allclose(working_point['firing_rates'],
                        network.results[self.prefix + 'firing_rates'])
        assert_allclose(working_point['mean_input'],
                        network.results[self.prefix + 'mean_input'])
        assert_allclose(working_point['std_input'],
                        network.results[self.prefix + 'std_input'])

    def test_delay_dist_matrix(self, network):
        delay_dist_matrix = nnmt.network_properties.delay_dist_matrix(network)
        assert_allclose(delay_dist_matrix, network.results['D'])

    def test_transfer_function(self, network):
        nnmt.lif.exp.working_point(network)
        transfer_function_taylor = nnmt.lif.exp.transfer_function(
            network, method='taylor')
        assert_allclose(transfer_function_taylor,
                        network.results[self.prefix + 'transfer_function'])
        transfer_function_shift = nnmt.lif.exp.transfer_function(
            network, method='shift')
        assert_allclose(transfer_function_shift,
                        network.results[self.prefix + 'transfer_function'])

    def test_transfer_function_single(self, network):
        nnmt.lif.exp.working_point(network)
        omega = network.analysis_params['omega']
        transfer_function_taylor = nnmt.lif.exp.transfer_function(
            network, omega, method='taylor')
        assert_allclose(
            transfer_function_taylor,
            network.results[self.prefix + 'transfer_function']
            )
        transfer_function_shift = nnmt.lif.exp.transfer_function(
            network, omega, method='shift')
        assert_allclose(
            transfer_function_shift,
            network.results[self.prefix + 'transfer_function']
            )

    def test_sensitivity_measure(self, network):
        nnmt.lif.exp.working_point(network)
        nnmt.lif.exp.transfer_function(network)
        nnmt.network_properties.delay_dist_matrix(network)
        nnmt.lif.exp.effective_connectivity(network)
        frequency = network.analysis_params['omega']/(2*np.pi)
        sm = nnmt.lif.exp.sensitivity_measure(network,
                                              frequency=frequency)
        sm_fix = network.results[self.prefix + 'sensitivity_measure']
        check_quantity_dicts_are_allclose(sm, sm_fix)

    def test_sensitivity_measure_all_eigenmodes(self, network):
        nnmt.lif.exp.working_point(network)
        nnmt.lif.exp.transfer_function(network)
        nnmt.network_properties.delay_dist_matrix(network)
        nnmt.lif.exp.effective_connectivity(network)
        margin = network.analysis_params['margin']
        sm = nnmt.lif.exp.sensitivity_measure_all_eigenmodes(
            network, margin=margin)
        sm_fix = network.results[
            self.prefix + 'sensitivity_measure_all_eigenmodes']
        check_quantity_dicts_are_allclose(sm, sm_fix)


    def test_power_spectra(self, network):
        nnmt.lif.exp.working_point(network)
        nnmt.lif.exp.transfer_function(network)
        nnmt.network_properties.delay_dist_matrix(network)
        nnmt.lif.exp.effective_connectivity(network)
        power_spectra = nnmt.lif.exp.power_spectra(network)
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
        nnmt.lif.exp.working_point(network)
        transfer_function_taylor_0 = nnmt.lif.exp.transfer_function(
            network, method=method)
        transfer_function_taylor_1 = nnmt.lif.exp.transfer_function(
            network, method=method)
        assert_quantity_allclose(transfer_function_taylor_0,
                                 transfer_function_taylor_1)

    def test_transfer_function_first_taylor_then_shift(self, network):
        nnmt.lif.exp.working_point(network)
        transfer_function_taylor = nnmt.lif.exp.transfer_function(
            network, method='taylor')
        transfer_function_shift = nnmt.lif.exp.transfer_function(
            network, method='shift')
        with pytest.raises(AssertionError):
            assert_quantity_allclose(transfer_function_taylor,
                                     transfer_function_shift)

    def test_transfer_function_single_first_taylor_then_shift(self, network):
        omega = network.analysis_params['omega']
        nnmt.lif.exp.working_point(network)
        tf_taylor = nnmt.lif.exp.transfer_function(
            network, omega, method='taylor')
        tf_shift = nnmt.lif.exp.transfer_function(
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
        network = nnmt.models.Microcircuit(negative_rate_params_file,
                                          analysis_params_file)
        firing_rates = nnmt.lif.exp.firing_rates(network)
        assert not any(firing_rates < 0)

class Test_ambiguous_match_eigenvalues_across_frequencies:
    """
    The matching of eigenvalues across frequencies can be ambiguous if
    two eigenvalues at frequency step i+1 are closest to the same eigenvalue
    at frequency step i. We here test that a warning is raised in such a
    situation.
    """

    def test_warning_ambiguous_match_eigenvalues(self):
        ambiguous_eigenvalues_params_file = (
            'tests/fixtures/integration/config/'
            'minimal_ambiguous_eigenvalues.yaml')
        ambiguous_eigenvalues_params = (
            nnmt.input_output.load_val_unit_dict_from_yaml(
                ambiguous_eigenvalues_params_file))
        print(ambiguous_eigenvalues_params)

        with pytest.warns(UserWarning):
            nnmt.lif.exp._match_eigenvalues_across_frequencies(
                **ambiguous_eigenvalues_params
            )
