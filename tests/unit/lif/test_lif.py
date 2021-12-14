import pytest
import numpy as np

from ...checks import (
    check_pos_params_neg_raise_exception)

from numpy.testing import (
    assert_array_equal,
    assert_allclose,
    )

import nnmt.lif._general as general
from nnmt import ureg


class Test_mean_input:

    func = staticmethod(general._mean_input)
    fixtures = 'lif_mean_input.h5'

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        assert_allclose(self.func(**params), output)


class Test_std_input:

    func = staticmethod(general._std_input)
    fixtures = 'lif_std_input.h5'

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        assert_allclose(self.func(**params), output)


class Test_input_function_wrapper:

    func = staticmethod(general._input_calc)

    def mock_mean_input(self, mocker):
        mocker.patch('nnmt.lif._general._mean_input',
                     return_value=1
                     )

    def test_raise_exception_if_not_all_parameters_available(self, mocker,
                                                             empty_network):
        self.mock_mean_input(mocker)
        empty_network.results['test.firing_rates'] = np.array([1]) * ureg.Hz
        with pytest.raises(RuntimeError):
            self.func(empty_network, 'test.', general._mean_input)

    def test_raise_exception_if_no_firing_rate_available(self, mocker,
                                                         empty_network):
        self.mock_mean_input(mocker)
        empty_network.network_params['tau_m'] = 1
        empty_network.network_params['K'] = 1
        empty_network.network_params['J'] = 1
        empty_network.network_params['K_ext'] = 1
        empty_network.network_params['J_ext'] = 1
        empty_network.network_params['nu_ext'] = 1
        with pytest.raises(RuntimeError):
            self.func(empty_network, 'test.', general._mean_input)

    def test_returns_unit(self, mocker, empty_network):
        self.mock_mean_input(mocker)
        empty_network.network_params['tau_m'] = 1
        empty_network.network_params['K'] = 1
        empty_network.network_params['J'] = 1
        empty_network.network_params['K_ext'] = 1
        empty_network.network_params['J_ext'] = 1
        empty_network.network_params['nu_ext'] = 1
        empty_network.results['test.firing_rates'] = np.array([1]) * ureg.Hz
        result = self.func(empty_network, 'test.', general._mean_input)
        assert isinstance(result, ureg.Quantity)


class Test_fit_transfer_function:

    func = staticmethod(general._fit_transfer_function)

    def test_fits_low_pass_filter_correctly(self):

        def low_pass_filter(omega, tau, h0):
            return h0 / (1. + 1j * omega * tau)

        tau = 0.01
        h0 = 0.8
        omegas = np.linspace(0, 500, 2000)
        transfunc = low_pass_filter(omegas, tau, h0)[np.newaxis].T

        transfunc_fit, tau_fit, h0_fit, fit_error = (
            general._fit_transfer_function(transfunc, omegas))

        assert tau_fit[0] == tau
        assert h0_fit[0] == h0
        assert_array_equal(transfunc_fit, transfunc)
        assert fit_error == 0
