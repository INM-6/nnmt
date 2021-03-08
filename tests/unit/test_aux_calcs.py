import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.special import erf, zetac
from scipy.integrate import quad

from .checks import (assert_units_equal,
                     check_pos_params_neg_raise_exception,
                     check_correct_output_for_several_mus_and_sigmas,
                     check_almost_correct_output_for_several_mus_and_sigmas,
                     check_V_0_larger_V_th_raise_exception,
                     check_warning_is_given_if_k_is_critical,
                     check_exception_is_raised_if_k_is_too_large)

import lif_meanfield_tools as lmt
from lif_meanfield_tools.aux_calcs import (
    siegert1,
    siegert2,
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
    determinant,
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
              * quad(integrand, y_r, y_th)[0])

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


class Test_siegert1:

    func = staticmethod(siegert1)
    decimal = 6

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params, pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_params)

    def test_mu_larger_V_th_raises_exception(self, std_params):
        std_params['mu'] = 1.1 * std_params['V_th_rel']
        with pytest.raises(ValueError):
            self.func(**std_params)

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        if any(params['mu'] > params['V_th_rel']):
            pytest.skip("Parameters out of range the function is intended "
                        "for.")
        check_almost_correct_output_for_several_mus_and_sigmas(
            self.func, real_siegert, params, self.decimal)


class Test_siegert2:

    func = staticmethod(siegert2)
    decimal = 6

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params, pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_params)

    def test_mu_smaller_V_th_raises_exception(self, std_params):
        std_params['mu'] = 0.9 * std_params['V_th_rel']
        with pytest.raises(ValueError):
            self.func(**std_params)

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        mus = params.pop('mu')
        sigmas = params.pop('sigma')
        for mu, sigma in zip(mus, sigmas):
            params['mu'] = mu
            params['sigma'] = sigma
            if mu > 0.95 * params['V_th_rel']:
                expected = real_siegert(**params)
                result = self.func(**params)
                assert_array_almost_equal(expected, result, self.decimal)
                assert_units_equal(expected, result)
            else:
                with pytest.raises(ValueError):
                    self.func(**params)


class Test_nu0_fb433:

    func = staticmethod(nu0_fb433)
    decimal = 6

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

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        check_almost_correct_output_for_several_mus_and_sigmas(
            self.func, real_shifted_siegert, params, self.decimal)


class Test_nu0_fb:

    func = staticmethod(nu0_fb)
    decimal = 6

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

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        check_almost_correct_output_for_several_mus_and_sigmas(
            self.func, real_shifted_siegert, params, self.decimal)


class Test_nu_0:

    func = staticmethod(nu_0)

    def test_sieger1_is_called_if_mu_smaller_V_th_rel(self, mocker,
                                                      std_params):
        std_params['mu'] = std_params['V_th_rel'] * 0.9
        mock = mocker.patch('lif_meanfield_tools.aux_calcs.siegert1')
        self.func(**std_params)
        mock.assert_called_once()

    def test_sieger2_is_called_if_mu_bigger_V_th_rel(self, mocker,
                                                     std_params):
        std_params['mu'] = std_params['V_th_rel'] * 1.1
        mock = mocker.patch('lif_meanfield_tools.aux_calcs.siegert2')
        self.func(**std_params)
        mock.assert_called_once()


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
    output_key = 'd_nu_d_nu_in_fb'

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

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        outputs = output_test_fixtures.pop('output')
        check_correct_output_for_several_mus_and_sigmas(self.func, params,
                                                        outputs)


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


@pytest.mark.xfail
class Test_determinant:

    func = staticmethod(determinant)

    def test_real_matrix_with_zero_determinant(self):
        a = [1, 2, 3]
        M = np.array([a, a, a])
        result = self.func(M)
        real_determinant = 0
        assert result == real_determinant

    def test_real_matrix_with_positive_determinant(self):
        M = np.array([[1, 2, 3], [2, 1, 3], [3, 1, 2]])
        result = self.func(M)
        real_determinant = 6
        assert result == real_determinant

    def test_real_matrix_with_negative_determinant(self):
        M = np.array([[1, 2, 3], [3, 1, 2], [2, 1, 3]])
        result = self.func(M)
        real_determinant = -6
        assert result == real_determinant

    def test_non_square_matrix(self):
        M = np.array([[1, 2, 3], [2, 3, 1]])
        with pytest.raises(np.linalg.LinAlgError):
            self.func(M)

    def test_matrix_with_imaginary_determinant(self):
        M = np.array([[complex(0, 1), 1], [0, 1]])
        real_determinant = np.linalg.det(M)
        result = self.func(M)
        assert result == real_determinant


class Test_determinant_same_rows:

    @pytest.mark.xfail
    def test_something(self):
        pass


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


class Test_solve_chareq_rate_boxcar:

    @pytest.mark.xfail
    def test_something(self):
        pass
