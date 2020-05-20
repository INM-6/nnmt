import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.special import erf, zetac
from scipy.integrate import quad

from .checks import (check_pos_params_neg_raise_exception,
                     check_correct_output,
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
    )

ureg = lmt.ureg

fixture_path = 'tests/fixtures/'


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
    
    V_th_eff = V_th_rel + sigma * alpha * k / 2
    V_0_eff = V_0_rel + sigma * alpha * k / 2
    
    nu = real_siegert(tau_m, tau_r, V_th_eff, V_0_eff, mu, sigma)
    
    return nu


class Test_siegert1:
    
    func = staticmethod(siegert1)
    precision = 10**-7
    
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
        check_almost_correct_output_for_several_mus_and_sigmas(
            self.func, real_siegert, params, self.precision)


class Test_siegert2:
    
    func = staticmethod(siegert2)
    precision = 10**-7
    
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
                assert_array_almost_equal(expected, result, self.precision)
            else:
                with pytest.raises(ValueError):
                    self.func(**params)
                    

class Test_nu0_fb433:
    
    func = staticmethod(nu0_fb433)
    precision = 10**-7
    
    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params, pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_params)

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        check_almost_correct_output_for_several_mus_and_sigmas(
            self.func, real_shifted_siegert, params, self.precision)

    
class Test_nu0_fb:
    
    func = staticmethod(nu0_fb)
    precision = 10**-7
    
    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params, pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_params)

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        check_almost_correct_output_for_several_mus_and_sigmas(
            self.func, real_shifted_siegert, params, self.precision)


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
                
    def test_zero_sigma_raises_error(self, std_params):
        std_params['sigma'] = 0
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
        
    def test_zero_sigma_raises_error(self, std_params):
        std_params['sigma'] = 0
        with pytest.raises(ZeroDivisionError):
            self.func(**std_params)
            
    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        outputs = output_test_fixtures.pop('output')
        check_correct_output_for_several_mus_and_sigmas(self.func, params,
                                                        outputs)
    
