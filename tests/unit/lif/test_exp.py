import pytest
import numpy as np
from scipy.special import zetac

from numpy.testing import (
    assert_allclose,
    )

from ...checks import (check_pos_params_neg_raise_exception,
                       check_V_0_larger_V_th_raise_exception,
                       check_warning_is_given_if_k_is_critical,
                       check_correct_output,
                       check_correct_output_for_several_mus_and_sigmas,
                       check_quantity_dicts_are_allclose)

from .test_delta import real_siegert

import nnmt
import nnmt.lif.exp as exp

from nnmt.utils import (
    _strip_units,
    _to_si_units,
    _convert_to_si_and_strip_units
    )


ureg = nnmt.ureg

fixture_path = 'tests/fixtures/unit/data/'


def real_shifted_siegert(tau_m, tau_s, tau_r,
                         V_th_rel, V_0_rel,
                         mu, sigma):
    """
    Siegert formula with shifted boundaries for the colored noise case.

    Introduced in Fourcaud 2002, and Schuecker 2015.
    """

    alpha = np.sqrt(2.) * abs(zetac(0.5) + 1)
    k = np.sqrt(tau_s / tau_m)

    V_th_eff = V_th_rel + sigma * alpha * k / 2
    V_0_eff = V_0_rel + sigma * alpha * k / 2

    nu = real_siegert(tau_m, tau_r, V_th_eff, V_0_eff, mu, sigma)

    return nu


@pytest.mark.old
class Test_firing_rates_old:

    func = staticmethod(exp._firing_rate_shift)
    output_key = 'firing_rates'

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params, pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_params):
        _strip_units(std_params)
        check_V_0_larger_V_th_raise_exception(self.func, std_params)

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        output = output_test_fixtures.pop('output')
        _strip_units(params)
        output = output.magnitude / 1000
        check_correct_output(self.func, params, output)


class Test_firing_rates_wrapper:

    func = staticmethod(exp.firing_rates)

    def mock_firing_rate_integration(self, mocker):
        mocker.patch(
            'nnmt.lif._general._firing_rate_integration',
            return_value=1
            )

    def test_raise_exception_if_not_all_parameters_available(self, mocker,
                                                             empty_network):
        self.mock_firing_rate_integration(mocker)
        with pytest.raises(RuntimeError):
            self.func(empty_network)


class Test_firing_rate_shift:

    func = staticmethod(exp._firing_rate_shift)
    fixtures = 'lif_exp_firing_rate_shift.h5'
    rtol = 1e-7

    def test_pos_params_neg_raise_exception(self, std_unitless_params,
                                            pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_unitless_params,
                                             pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_unitless_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_unitless_params)

    def test_warning_is_given_if_k_is_critical(self, std_unitless_params):
        check_warning_is_given_if_k_is_critical(self.func, std_unitless_params)

    def test_gives_similar_results_as_real_shifted_siegert_if_it_converges(
            self, unit_fixtures):
        params = unit_fixtures.pop('params')
        siegert = real_shifted_siegert(**params)
        if not np.any(np.isnan(siegert)):
            assert_allclose(self.func(**params), siegert, rtol=self.rtol)

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        assert_allclose(self.func(**params), output)

    def test_get_same_results_vectorized(self, unit_fixtures_fully_vectorized):
        params = unit_fixtures_fully_vectorized.pop('params')
        vectorized_output = self.func(**params)
        single_outputs = np.zeros(vectorized_output.shape)
        mus = params.pop('mu')
        sigmas = params.pop('sigma')
        tau_ms = params.pop('tau_m')
        tau_rs = params.pop('tau_r')
        tau_ss = params.pop('tau_s')
        V_th_rels = params.pop('V_th_rel')
        V_0_rels = params.pop('V_0_rel')
        for j, (mu, sigma, tau_m, tau_r, tau_s, V_th_rel, V_0_rel
                ) in enumerate(
                    zip(mus, sigmas, tau_ms, tau_rs, tau_ss, V_th_rels,
                        V_0_rels)):
            single_outputs[j] = self.func(mu=mu, sigma=sigma,
                                          tau_m=tau_m, tau_s=tau_s,
                                          tau_r=tau_r,
                                          V_th_rel=V_th_rel,
                                          V_0_rel=V_0_rel)
        assert_allclose(vectorized_output, single_outputs)


class Test_firing_rate_taylor:

    func = staticmethod(exp._firing_rate_taylor)
    fixtures = 'lif_exp_firing_rate_taylor.h5'
    # Lower rtol than for nu0_fb because it is compared to real_shifted_siegert
    # instead of the corresponding Taylor approximation.
    rtol = 0.2

    def test_pos_params_neg_raise_exception(self, std_unitless_params,
                                            pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_unitless_params,
                                             pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_unitless_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_unitless_params)

    def test_warning_is_given_if_k_is_critical(self, std_unitless_params):
        check_warning_is_given_if_k_is_critical(self.func, std_unitless_params)

    def test_gives_similar_results_as_real_shifted_siegert_if_it_converges(
            self, unit_fixtures):
        params = unit_fixtures.pop('params')
        siegert = real_shifted_siegert(**params)
        result = self.func(**params)
        if not np.any(result < 0):
            if not np.any(np.isnan(siegert)):
                assert_allclose(result, siegert, rtol=self.rtol)
            else:
                pytest.skip('Shifted Siegert did not converge.')
        else:
            pytest.skip('Negative rates detected.')

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        assert_allclose(self.func(**params), output)

    def test_get_same_results_vectorized(self, unit_fixtures_fully_vectorized):
        params = unit_fixtures_fully_vectorized.pop('params')
        vectorized_output = self.func(**params)
        single_outputs = np.zeros(vectorized_output.shape)
        mus = params.pop('mu')
        sigmas = params.pop('sigma')
        tau_ms = params.pop('tau_m')
        tau_rs = params.pop('tau_r')
        tau_ss = params.pop('tau_s')
        V_th_rels = params.pop('V_th_rel')
        V_0_rels = params.pop('V_0_rel')
        for j, (mu, sigma, tau_m, tau_r, tau_s, V_th_rel, V_0_rel
                ) in enumerate(
                    zip(mus, sigmas, tau_ms, tau_rs, tau_ss, V_th_rels,
                        V_0_rels)):
            single_outputs[j] = self.func(mu=mu, sigma=sigma,
                                          tau_m=tau_m, tau_s=tau_s,
                                          tau_r=tau_r,
                                          V_th_rel=V_th_rel,
                                          V_0_rel=V_0_rel)
        assert_allclose(vectorized_output, single_outputs)


class Test_Phi:

    func = staticmethod(exp._Phi)

    def test_correct_output(self):
        fixtures = np.load(fixture_path + 'Phi.npz')
        s_values = fixtures['s_values']
        outputs = fixtures['outputs']
        for s, output in zip(s_values, outputs):
            result = self.func(s)
            assert result == output


class Test_mean_input:

    func = staticmethod(exp._mean_input)
    fixtures = 'lif_mean_input.h5'

    def test_pos_params_neg_raise_exception(self, std_unitless_params,
                                            pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_unitless_params,
                                             pos_keys)

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        assert_allclose(self.func(**params), output)


class Test_mean_input_wrapper:

    func = staticmethod(exp.mean_input)

    def mock_mean_input(self, mocker):
        mocker.patch(
            'nnmt.lif.exp._mean_input',
            return_value=1
            )

    def test_raise_exception_if_not_all_parameters_available(self, mocker,
                                                             empty_network):
        self.mock_mean_input(mocker)
        empty_network.results['lif.exp.firing_rates'] = np.array([1]) * ureg.Hz
        with pytest.raises(RuntimeError):
            self.func(empty_network)

    def test_raise_exception_if_rates_not_available(self, mocker,
                                                    empty_network):

        self.mock_mean_input(mocker)
        empty_network.network_params['tau_m'] = 1
        empty_network.network_params['K'] = 1
        empty_network.network_params['J'] = 1
        empty_network.network_params['K_ext'] = 1
        empty_network.network_params['J_ext'] = 1
        empty_network.network_params['nu_ext'] = 1
        with pytest.raises(RuntimeError):
            self.func(empty_network)


class Test_std_input:

    func = staticmethod(exp._std_input)
    fixtures = 'lif_std_input.h5'

    def test_pos_params_neg_raise_exception(self, std_unitless_params,
                                            pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_unitless_params,
                                             pos_keys)

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        assert_allclose(self.func(**params), output)


class Test_std_input_wrapper:

    func = staticmethod(exp.std_input)

    def mock_std_input(self, mocker):
        mocker.patch(
            'nnmt.lif.exp._std_input',
            return_value=1
            )

    def test_raise_exception_if_not_all_parameters_available(self, mocker,
                                                             empty_network):
        self.mock_std_input(mocker)
        empty_network.results['lif.exp.firing_rates'] = np.array([1]) * ureg.Hz
        with pytest.raises(RuntimeError):
            self.func(empty_network)

    def test_raise_exception_if_rates_not_available(self, mocker,
                                                    empty_network):

        self.mock_std_input(mocker)
        empty_network.network_params['tau_m'] = 1
        empty_network.network_params['K'] = 1
        empty_network.network_params['J'] = 1
        empty_network.network_params['K_ext'] = 1
        empty_network.network_params['J_ext'] = 1
        empty_network.network_params['nu_ext'] = 1
        with pytest.raises(RuntimeError):
            self.func(empty_network)


@pytest.mark.old
class Test_transfer_function_shift_old():

    func = staticmethod(exp._transfer_function_shift)
    output_key = 'tf_shift'

    # def test_pos_params_neg_raise_exception(self, std_params_tf, pos_keys):
    #     check_pos_params_neg_raise_exception(self.func, std_params_tf,
    #                                          pos_keys)
    #
    # def test_warning_is_given_if_k_is_critical(self, std_params_tf):
    #     check_warning_is_given_if_k_is_critical(self.func, std_params_tf)
    #
    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        output = output_test_fixtures.pop('output')
        output = output.magnitude * 1000
        _to_si_units(params)
        _strip_units(params)
        assert_allclose(self.func(**params), output)


@pytest.mark.old
class Test_transfer_function_taylor_old():

    func = staticmethod(exp._transfer_function_taylor)
    output_key = 'tf_taylor'

    # def test_pos_params_neg_raise_exception(self, std_params_tf, pos_keys):
    #     check_pos_params_neg_raise_exception(self.func, std_params_tf,
    #                                          pos_keys)
    #
    # def test_warning_is_given_if_k_is_critical(self, std_params_tf):
    #     check_warning_is_given_if_k_is_critical(self.func, std_params_tf)
    #
    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        output = output_test_fixtures.pop('output')
        output = output.magnitude * 1000
        _to_si_units(params)
        _strip_units(params)
        assert_allclose(self.func(**params), output)


class Test_Phi_prime_mu:

    func = staticmethod(exp._Phi_prime_mu)

    def test_negative_sigma_raises_error(self):
        sigma = -1 * ureg.mV
        s = 1
        with pytest.raises(ValueError):
            self.func(s, sigma)

    def test_correct_output(self):
        fixtures = np.load(fixture_path + 'Phi_prime_mu.npz')
        s_values = fixtures['s_values']
        sigmas = fixtures['sigmas']
        outputs = fixtures['outputs']
        for s, sigma, output in zip(s_values, sigmas, outputs):
            result = self.func(s, sigma)
            assert result == output

    def test_zero_sigma_raises_error(self):
        sigma = 0 * ureg.mV
        s = 1
        with pytest.raises(ZeroDivisionError):
            self.func(s, sigma)


class Test_transfer_function_shift:

    func = staticmethod(exp._transfer_function_shift)
    fixtures = 'lif_exp_transfer_function_shift.h5'
    rtol = 1e-7

    def test_pos_params_neg_raise_exception(self, std_unitless_params,
                                            pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_unitless_params,
                                             pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_unitless_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_unitless_params)

    def test_warning_is_given_if_k_is_critical(self, std_unitless_params):
        check_warning_is_given_if_k_is_critical(self.func, std_unitless_params)

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        assert_allclose(self.func(**params), output)

    def test_get_same_results_vectorized(self, unit_fixtures_fully_vectorized):
        params = unit_fixtures_fully_vectorized.pop('params')
        vectorized_output = self.func(**params)
        single_outputs = np.zeros(vectorized_output.shape, dtype=complex)
        mus = params.pop('mu')
        sigmas = params.pop('sigma')
        tau_ms = params.pop('tau_m')
        tau_rs = params.pop('tau_r')
        tau_ss = params.pop('tau_s')
        V_th_rels = params.pop('V_th_rel')
        V_0_rels = params.pop('V_0_rel')
        for i, omega in enumerate(params.pop('omegas')):
            for j, (mu, sigma, tau_m, tau_r, tau_s, V_th_rel, V_0_rel
                    ) in enumerate(
                    zip(mus, sigmas, tau_ms, tau_rs, tau_ss, V_th_rels,
                        V_0_rels)):
                single_outputs[i, j] = self.func(mu=mu, sigma=sigma,
                                                 omegas=omega, tau_m=tau_m,
                                                 tau_s=tau_s, tau_r=tau_r,
                                                 V_th_rel=V_th_rel,
                                                 V_0_rel=V_0_rel)
        assert_allclose(vectorized_output, single_outputs)


class Test_transfer_function_taylor:

    func = staticmethod(exp._transfer_function_taylor)
    fixtures = 'lif_exp_transfer_function_taylor.h5'
    rtol = 1e-7

    def test_pos_params_neg_raise_exception(self, std_unitless_params,
                                            pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_unitless_params,
                                             pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_unitless_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_unitless_params)

    def test_warning_is_given_if_k_is_critical(self, std_unitless_params):
        check_warning_is_given_if_k_is_critical(self.func, std_unitless_params)

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        assert_allclose(self.func(**params), output)

    def test_get_same_results_vectorized(self, unit_fixtures_fully_vectorized):
        params = unit_fixtures_fully_vectorized.pop('params')
        vectorized_output = self.func(**params)
        single_outputs = np.zeros(vectorized_output.shape, dtype=complex)
        mus = params.pop('mu')
        sigmas = params.pop('sigma')
        tau_ms = params.pop('tau_m')
        tau_rs = params.pop('tau_r')
        tau_ss = params.pop('tau_s')
        V_th_rels = params.pop('V_th_rel')
        V_0_rels = params.pop('V_0_rel')
        for i, omega in enumerate(params.pop('omegas')):
            for j, (mu, sigma, tau_m, tau_r, tau_s, V_th_rel, V_0_rel
                    ) in enumerate(
                    zip(mus, sigmas, tau_ms, tau_rs, tau_ss, V_th_rels,
                        V_0_rels)):
                single_outputs[i, j] = self.func(mu=mu, sigma=sigma,
                                                 omegas=omega, tau_m=tau_m,
                                                 tau_s=tau_s, tau_r=tau_r,
                                                 V_th_rel=V_th_rel,
                                                 V_0_rel=V_0_rel)
        assert_allclose(vectorized_output, single_outputs)


class Test_derivative_of_firing_rates_wrt_mean_input:

    func = staticmethod(exp._derivative_of_firing_rates_wrt_mean_input)
    output_key = 'd_nu_d_mu_fb433'
    fixtures = 'lif_exp_derivative_of_firing_rates_wrt_mean_input.h5'

    def test_pos_params_neg_raise_exception(self, std_unitless_params,
                                            pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_unitless_params,
                                             pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_unitless_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_unitless_params)

    def test_warning_is_given_if_k_is_critical(self, std_unitless_params):
        check_warning_is_given_if_k_is_critical(self.func, std_unitless_params)

    def test_zero_sigma_raises_error(self, std_params):
        std_params['sigma'] = 0 * ureg.mV
        with pytest.raises(ZeroDivisionError):
            self.func(**std_params)

    def test_correct_output_old(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        outputs = output_test_fixtures.pop('output')
        _to_si_units(params)
        _strip_units(params)
        outputs = outputs.magnitude * 1000
        check_correct_output_for_several_mus_and_sigmas(self.func, params,
                                                        outputs)

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        assert_allclose(self.func(**params), output)


class Test_Psi:

    func = staticmethod(exp._Psi)

    def test_correct_output(self, mocker):
        fixtures = np.load(fixture_path + 'Psi.npz')
        zs = fixtures['zs']
        xs = fixtures['xs']
        pcfus = fixtures['pcfus']
        outputs = fixtures['outputs']
        mock = mocker.patch('nnmt.lif.exp.pcfu_vec')
        mock.side_effect = pcfus
        for z, x, output in zip(zs, xs, outputs):
            result = self.func(z, x)
            assert result == output


class Test_d_Psi:

    func = staticmethod(exp._d_Psi)

    def test_correct_output(self, mocker):
        fixtures = np.load(fixture_path + 'd_Psi.npz')
        zs = fixtures['zs']
        xs = fixtures['xs']
        psis = fixtures['psis']
        outputs = fixtures['outputs']
        mock = mocker.patch('nnmt.lif.exp._Psi')
        mock.side_effect = psis
        for z, x, output in zip(zs, xs, outputs):
            result = self.func(z, x)
            assert result == output


class Test_d_2_Psi:

    func = staticmethod(exp._d_2_Psi)

    def test_correct_output(self, mocker):
        fixtures = np.load(fixture_path + 'd_2_Psi.npz')
        zs = fixtures['zs']
        xs = fixtures['xs']
        psis = fixtures['psis']
        outputs = fixtures['outputs']
        mock = mocker.patch('nnmt.lif.exp._Psi')
        mock.side_effect = psis
        for z, x, output in zip(zs, xs, outputs):
            result = self.func(z, x)
            assert result == output


class Test_derivative_of_firing_rates_wrt_input_rate:

    func = staticmethod(exp._derivative_of_firing_rates_wrt_input_rate)
    output_keys = ['d_nu_d_nu_in_fb_all']
    fixtures = 'lif_exp_derivative_of_firing_rates_wrt_input_rate.h5'

    def test_pos_params_neg_raise_exception(self, std_unitless_params,
                                            pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_unitless_params,
                                             pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_unitless_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_unitless_params)

    def test_warning_is_given_if_k_is_critical(self, std_unitless_params):
        check_warning_is_given_if_k_is_critical(self.func, std_unitless_params)

    def test_zero_sigma_raises_error(self, std_params):
        std_params['sigma'] = 0
        with pytest.raises(ZeroDivisionError):
            self.func(**std_params)

    def test_correct_output_old(self, output_test_fixtures):
        params = output_test_fixtures['params']
        _convert_to_si_and_strip_units(params)
        result = self.func(**params)
        # I suppose there was a mistake in the original function multiplying
        # tau_m in ms with nu0 in Hz, thereby adding another factor 1000000
        # because they appear squared.
        output = output_test_fixtures['output'][0] / 1000000
        assert_allclose(result, output)

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        assert_allclose(self.func(**params), output)


class Test_effective_connectivity:

    func = staticmethod(exp._effective_connectivity)
    fixtures = 'lif_exp_effective_connectivity.h5'

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        assert_allclose(self.func(**params), output)


class Test_propagator:

    func = staticmethod(exp._propagator)
    fixtures = 'lif_exp_propagator.h5'

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        assert_allclose(self.func(**params), output)


class Test_sensitivity_measure:

    func = staticmethod(exp._sensitivity_measure)
    fixtures = 'lif_exp_sensitivity_measure.h5'

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        output_func = self.func(**params)

        check_quantity_dicts_are_allclose(output, output_func)


class Test_sensitivity_measure_all_eigenmodes:

    func = staticmethod(exp._sensitivity_measure_all_eigenmodes)
    fixtures = 'lif_exp_sensitivity_measure_all_eigenmodes.h5'

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        output_func = self.func(**params)

        check_quantity_dicts_are_allclose(output, output_func)


class Test_match_eigenvalues_across_frequencies:

    func = staticmethod(exp._match_eigenvalues_across_frequencies)
    fixtures = 'lif_exp_match_eigenvalues_across_frequencies.h5'

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')

        assert_allclose(self.func(**params), output)


class Test_power_spectra:

    func = staticmethod(exp._power_spectra)
    fixtures = 'lif_exp_power_spectra.h5'

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params,
                                             pos_keys)

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        assert_allclose(self.func(**params), output)


class Test_external_rates_for_fixed_input:

    func = staticmethod(exp._external_rates_for_fixed_input)
    fixtures = 'lif_exp_external_rates_for_fixed_input.h5'

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params,
                                             pos_keys)

    def test_negative_rates_raise_exception(self, unit_fixtures, mocker):
        params = unit_fixtures.pop('params')
        mocker.patch(
            'nnmt.lif.exp.np.linalg.lstsq',
            return_value=np.array([[10, 20, -1],
                                   [10, 20, -1],
                                   [10, 20, -1]]))
        with pytest.raises(RuntimeError):
            self.func(**params)

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        assert_allclose(self.func(**params), output)
