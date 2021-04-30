import pytest
import numpy as np
from scipy.integrate import quad
from scipy.special import erf, erfcx, zetac


from ...checks import (check_pos_params_neg_raise_exception,
                       check_correct_output_for_several_mus_and_sigmas,
                       check_almost_correct_output_for_several_mus_and_sigmas,
                       check_V_0_larger_V_th_raise_exception,
                       check_warning_is_given_if_k_is_critical,
                       check_exception_is_raised_if_k_is_too_large
                       )

from .test_delta import real_siegert

import lif_meanfield_tools as lmt
import lif_meanfield_tools.lif.exp.static as exp
from lif_meanfield_tools.utils import _strip_units

ureg = lmt.ureg

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

    V_th_eff = V_th_rel + sigma * alpha * k / np.sqrt(2)
    V_0_eff = V_0_rel + sigma * alpha * k / np.sqrt(2)

    nu = real_siegert(tau_m, tau_r, V_th_eff, V_0_eff, mu, sigma)

    return nu


class Test_firing_rates_wrapper:
    
    func = staticmethod(exp.firing_rates)
    
    def mock_firing_rate_integration(self, mocker):
        mocker.patch(
            'lif_meanfield_tools.lif.static._firing_rate_integration',
            return_value=1
            )
    
    def test_raise_exception_if_not_all_parameters_available(self, mocker,
                                                             empty_network):
        self.mock_firing_rate_integration(mocker)
        with pytest.raises(RuntimeError):
            self.func(empty_network)
        
    def test_returns_unit(self, mocker, network):
        self.mock_firing_rate_integration(mocker)
        result = self.func(network)
        assert isinstance(result, ureg.Quantity)
    

class Test_firing_rate_shift:
    
    func = staticmethod(exp._firing_rate_shift)
    rtol = 0.02

    def test_pos_params_neg_raise_exception(self, std_unitless_params,
                                            pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_unitless_params,
                                             pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_unitless_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_unitless_params)
        
    def test_warning_is_given_if_k_is_critical(self, std_unitless_params):
        check_warning_is_given_if_k_is_critical(self.func, std_unitless_params)

    def test_exception_is_raised_if_k_is_too_large(self, std_unitless_params):
        check_exception_is_raised_if_k_is_too_large(self.func,
                                                    std_unitless_params)

    def test_gives_similar_results_as_real_shifted_siegert(
            self, output_fixtures_mean_driven):
        params = output_fixtures_mean_driven.pop('params')
        _strip_units(params)
        check_almost_correct_output_for_several_mus_and_sigmas(
            self.func, real_shifted_siegert, params, self.rtol)

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        
        output = output_test_fixtures.pop('output')
        check_correct_output_for_several_mus_and_sigmas(
            self.func, params, output)


class Test_firing_rate_taylor:
    
    func = staticmethod(exp._firing_rate_taylor)
    rtol = 0.02

    def test_pos_params_neg_raise_exception(self, std_unitless_params,
                                            pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_unitless_params,
                                             pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_unitless_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_unitless_params)
        
    def test_warning_is_given_if_k_is_critical(self, std_unitless_params):
        check_warning_is_given_if_k_is_critical(self.func, std_unitless_params)

    def test_exception_is_raised_if_k_is_too_large(self, std_unitless_params):
        check_exception_is_raised_if_k_is_too_large(self.func,
                                                    std_unitless_params)

    def test_gives_similar_results_as_real_shifted_siegert(
            self, output_fixtures_mean_driven):
        params = output_fixtures_mean_driven.pop('params')
        _strip_units(params)
        check_almost_correct_output_for_several_mus_and_sigmas(
            self.func, real_shifted_siegert, params, self.rtol)

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        output = output_test_fixtures.pop('output')
        check_correct_output_for_several_mus_and_sigmas(
            self.func, params, output)


class Test_Phi:

    func = staticmethod(exp._Phi)

    def test_correct_output(self):
        fixtures = np.load(fixture_path + 'Phi.npz')
        s_values = fixtures['s_values']
        outputs = fixtures['outputs']
        for s, output in zip(s_values, outputs):
            result = self.func(s)
            assert result == output
