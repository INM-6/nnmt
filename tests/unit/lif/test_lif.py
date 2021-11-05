import pytest
import numpy as np

from ...checks import (
    check_pos_params_neg_raise_exception)

from numpy.testing import (
    assert_array_equal,
    )

import nnmt.lif._general as static
from nnmt import ureg


class Test_mean_input:

    func = staticmethod(static._mean_input)
    fixtures = 'lif_mean_input.h5'

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        print(output)
        assert_array_equal(self.func(**params), output)


class Test_std_input:

    func = staticmethod(static._std_input)
    fixtures = 'lif_std_input.h5'

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        print(output)

        assert_array_equal(self.func(**params), output)


class Test_input_function_wrapper:

    func = staticmethod(static._input_calc)

    def mock_mean_input(self, mocker):
        mocker.patch('nnmt.lif._general._mean_input',
                     return_value=1
                     )

    def test_raise_exception_if_not_all_parameters_available(self, mocker,
                                                             empty_network):
        self.mock_mean_input(mocker)
        empty_network.results['test.firing_rates'] = np.array([1]) * ureg.Hz
        with pytest.raises(RuntimeError):
            self.func(empty_network, 'test.', static._mean_input)

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
            self.func(empty_network, 'test.', static._mean_input)

    def test_returns_unit(self, mocker, empty_network):
        self.mock_mean_input(mocker)
        empty_network.network_params['tau_m'] = 1
        empty_network.network_params['K'] = 1
        empty_network.network_params['J'] = 1
        empty_network.network_params['K_ext'] = 1
        empty_network.network_params['J_ext'] = 1
        empty_network.network_params['nu_ext'] = 1
        empty_network.results['test.firing_rates'] = np.array([1]) * ureg.Hz
        result = self.func(empty_network, 'test.', static._mean_input)
        assert isinstance(result, ureg.Quantity)

