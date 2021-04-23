import pytest
import numpy as np
from scipy.special import erf, zetac
from scipy.integrate import quad

from ..checks import (assert_array_almost_equal,
                      assert_units_equal,
                      check_pos_params_neg_raise_exception,
                      check_correct_output_for_several_mus_and_sigmas,
                      check_almost_correct_output_for_several_mus_and_sigmas,
                      check_V_0_larger_V_th_raise_exception,
                      check_warning_is_given_if_k_is_critical,
                      check_exception_is_raised_if_k_is_too_large)

import lif_meanfield_tools as lmt
from lif_meanfield_tools.aux_calcs import (
    nu0_fb433,
    nu0_fb,
    nu_0,
    Phi,
    Phi_prime_mu,
    d_nu_d_mu,
    d_nu_d_mu_fb433,
    d_nu_d_nu_in_fb,
    Psi,
    d_Psi,
    d_2_Psi,
    p_hat_boxcar,
    )

ureg = lmt.ureg

fixture_path = 'tests/fixtures/unit/data/'


def strip_units_from_quantity_dict(d):
    """ Returns dictionary only containing magnitudes. """
    return {key: value.magnitude for key, value in d.items()}


def integrand(x):
    return np.exp(x**2) * (1 + erf(x))


def real_siegert(tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma):
    """ Siegert formula as given in Fourcaud Brunel 2002 eq. 4.11 """

    y_th = (V_th_rel - mu) / sigma
    y_r = (V_0_rel - mu) / sigma

    nu = 1 / (tau_r + np.sqrt(np.pi) * tau_m
              * quad(integrand, y_r, y_th, epsabs=1e-6)[0])

    return nu


def real_shifted_siegert(tau_m, tau_s, tau_r,
                         V_th_rel, V_0_rel,
                         mu, sigma):
    """
    Siegert formula with shifted boundaries for the colored noise case.

    Introduced in Fourcaud 2002, and Schuecker 2015.
    """

    alpha = np.sqrt(2.) * abs(zetac(0.5) + 1)
    k = np.sqrt(tau_s / tau_m)

    V_th_eff = V_th_rel + sigma * alpha * k / np.sqrt(2)
    V_0_eff = V_0_rel + sigma * alpha * k / np.sqrt(2)

    nu = real_siegert(tau_m, tau_r, V_th_eff, V_0_eff, mu, sigma)

    return nu


class Test_nu0_fb433:

    func = staticmethod(nu0_fb433)
    rtol = 0.05

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params, pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_params)

    def test_warning_is_given_if_k_is_critical(self,
                                               std_params_single_population):
        check_warning_is_given_if_k_is_critical(self.func,
                                                std_params_single_population)

    def test_exception_is_raised_if_k_is_too_large(self, std_params):
        check_exception_is_raised_if_k_is_too_large(self.func, std_params)

    def test_gives_similar_results_as_real_shifted_siegert(
            self, output_fixtures_mean_driven):
        params = output_fixtures_mean_driven.pop('params')
        check_almost_correct_output_for_several_mus_and_sigmas(
            self.func, real_shifted_siegert, params, self.rtol)
        
    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        output = output_test_fixtures.pop('output')
        check_correct_output_for_several_mus_and_sigmas(
            self.func, params, output)


class Test_nu0_fb:

    func = staticmethod(nu0_fb)
    rtol = 0.05
    
    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params, pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_params)

    def test_warning_is_given_if_k_is_critical(self,
                                               std_params_single_population):
        check_warning_is_given_if_k_is_critical(self.func,
                                                std_params_single_population)

    def test_exception_is_raised_if_k_is_too_large(self, std_params):
        check_exception_is_raised_if_k_is_too_large(self.func, std_params)

    def test_gives_similar_results_as_real_shifted_siegert(
            self, output_fixtures_mean_driven):
        params = output_fixtures_mean_driven.pop('params')
        check_almost_correct_output_for_several_mus_and_sigmas(
            self.func, real_shifted_siegert, params, self.rtol)
        
    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        output = output_test_fixtures.pop('output')
        check_correct_output_for_several_mus_and_sigmas(
            self.func, params, output)


class Test_Phi:

    func = staticmethod(Phi)

    def test_correct_output(self):
        fixtures = np.load(fixture_path + 'Phi.npz')
        s_values = fixtures['s_values']
        outputs = fixtures['outputs']
        for s, output in zip(s_values, outputs):
            result = self.func(s)
            assert result == output


class Test_Phi_prime_mu:

    func = staticmethod(Phi_prime_mu)

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


class Test_d_nu_d_mu:

    func = staticmethod(d_nu_d_mu)
    output_key = 'd_nu_d_mu'

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params, pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_params)

    def test_zero_sigma_raises_error(self, std_params):
        std_params['sigma'] = 0
        with pytest.raises(ZeroDivisionError):
            self.func(**std_params)

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        params = strip_units_from_quantity_dict(params)
        outputs = output_test_fixtures.pop('output')
        check_correct_output_for_several_mus_and_sigmas(self.func, params,
                                                        outputs)


class Test_d_nu_d_mu_fb433:

    func = staticmethod(d_nu_d_mu_fb433)
    output_key = 'd_nu_d_mu_fb433'

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params, pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_params)

    def test_warning_is_given_if_k_is_critical(self,
                                               std_params_single_population):
        check_warning_is_given_if_k_is_critical(self.func,
                                                std_params_single_population)

    def test_exception_is_raised_if_k_is_too_large(self, std_params):
        check_exception_is_raised_if_k_is_too_large(self.func, std_params)

    def test_zero_sigma_raises_error(self, std_params):
        std_params['sigma'] = 0 * ureg.mV
        with pytest.raises(ZeroDivisionError):
            self.func(**std_params)
        
    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        outputs = output_test_fixtures.pop('output')
        check_correct_output_for_several_mus_and_sigmas(self.func, params,
                                                        outputs)


class Test_d_nu_d_nu_in_fb:

    func = staticmethod(d_nu_d_nu_in_fb)
    output_keys = ['d_nu_d_nu_in_fb_mu', 'd_nu_d_nu_in_fb_sigma',
                   'd_nu_d_nu_in_fb_all']

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params, pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_params)

    def test_warning_is_given_if_k_is_critical(
            self, std_params_single_population):
        check_warning_is_given_if_k_is_critical(
            self.func, std_params_single_population)

    def test_exception_is_raised_if_k_is_too_large(self, std_params):
        check_exception_is_raised_if_k_is_too_large(self.func, std_params)

    def test_zero_sigma_raises_error(self, std_params):
        std_params['sigma'] = 0
        with pytest.raises(ZeroDivisionError):
            self.func(**std_params)

    @pytest.mark.parametrize('contribution, num', [('mu', 0), ('sigma', 1),
                                                   ('all', 2)])
    def test_correct_output(self, output_test_fixtures, contribution, num):
        params = output_test_fixtures['params']
        outputs = output_test_fixtures['output']
        params['contributions'] = contribution
        output = outputs[num]
        check_correct_output_for_several_mus_and_sigmas(self.func, params,
                                                        output)


class Test_Psi:

    func = staticmethod(Psi)

    def test_correct_output(self, mocker):
        fixtures = np.load(fixture_path + 'Psi.npz')
        zs = fixtures['zs']
        xs = fixtures['xs']
        pcfus = fixtures['pcfus']
        outputs = fixtures['outputs']
        mock = mocker.patch('lif_meanfield_tools.aux_calcs.mpmath.pcfu')
        mock.side_effect = pcfus
        for z, x, output in zip(zs, xs, outputs):
            result = self.func(z, x)
            assert result == output


class Test_d_Psi:

    func = staticmethod(d_Psi)

    def test_correct_output(self, mocker):
        fixtures = np.load(fixture_path + 'd_Psi.npz')
        zs = fixtures['zs']
        xs = fixtures['xs']
        psis = fixtures['psis']
        outputs = fixtures['outputs']
        mock = mocker.patch('lif_meanfield_tools.aux_calcs.Psi')
        mock.side_effect = psis
        for z, x, output in zip(zs, xs, outputs):
            result = self.func(z, x)
            assert result == output


class Test_d_2_Psi:

    func = staticmethod(d_2_Psi)

    def test_correct_output(self, mocker):
        fixtures = np.load(fixture_path + 'd_2_Psi.npz')
        zs = fixtures['zs']
        xs = fixtures['xs']
        psis = fixtures['psis']
        outputs = fixtures['outputs']
        mock = mocker.patch('lif_meanfield_tools.aux_calcs.Psi')
        mock.side_effect = psis
        for z, x, output in zip(zs, xs, outputs):
            result = self.func(z, x)
            assert result == output


class Test_p_hat_boxcar:

    func = staticmethod(p_hat_boxcar)

    def test_correct_output(self):
        fixtures = np.load(fixture_path + 'p_hat_boxcar.npz')
        ks = fixtures['ks']
        widths = fixtures['widths']
        outputs = fixtures['outputs']
        for z, x, output in zip(ks, widths, outputs):
            result = self.func(z, x)
            assert result == output
