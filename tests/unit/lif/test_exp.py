import pytest
import numpy as np
from scipy.special import zetac

from numpy.testing import (
    assert_allclose
    )

from ...checks import (check_pos_params_neg_raise_exception,
                       check_V_0_larger_V_th_raise_exception,
                       check_warning_is_given_if_k_is_critical,
                       check_correct_output)

from .test_delta import real_siegert

import lif_meanfield_tools as lmt
import lif_meanfield_tools.lif.exp as exp

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

    V_th_eff = V_th_rel + sigma * alpha * k / 2
    V_0_eff = V_0_rel + sigma * alpha * k / 2

    nu = real_siegert(tau_m, tau_r, V_th_eff, V_0_eff, mu, sigma)

    return nu


@pytest.mark.old
class Test_firing_rates:

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
        # _strip_units(output)
        output = output.magnitude / 1000
        check_correct_output(self.func, params, output)


class Test_firing_rates_wrapper:
    
    func = staticmethod(exp.firing_rates)
    
    def mock_firing_rate_integration(self, mocker):
        mocker.patch(
            'lif_meanfield_tools.lif._static._firing_rate_integration',
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
    
        
class Test_Phi:

    func = staticmethod(exp._Phi)

    def test_correct_output(self):
        fixtures = np.load(fixture_path + 'Phi.npz')
        s_values = fixtures['s_values']
        outputs = fixtures['outputs']
        for s, output in zip(s_values, outputs):
            result = self.func(s)
            assert result == output
