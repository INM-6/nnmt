import pytest
import numpy as np

from numpy.testing import assert_allclose

import nnmt.binary as binary

class Test_mean_activity_wrapper:

    func = staticmethod(binary.mean_activity)

    def mock_firing_rate_integration(self, mocker):
        mocker.patch(
            'nnmt._solvers._firing_rate_integration',
            return_value=1
            )

    def test_raise_exception_if_not_all_parameters_available(self, mocker,
                                                             empty_network):
        self.mock_firing_rate_integration(mocker)
        with pytest.raises(RuntimeError):
            self.func(empty_network)


class Test_mean_activity_for_given_input:

    func = staticmethod(binary._mean_activity_for_given_input)
    fixtures = 'binary_mean_activity.h5'
    rtol = 1e-7

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        assert_allclose(self.func(**params), output)


class Test_mean_input:

    func = staticmethod(binary._mean_input)
    fixtures = 'binary_mean_input.h5'

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        assert_allclose(self.func(**params), output)

class Test_std_input:

    func = staticmethod(binary._std_input)
    fixtures = 'binary_std_input.h5'

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        assert_allclose(self.func(**params), output)

class Test_balanced_threshold:

    func = staticmethod(binary._mean_input)
    fixtures = 'binary_balanced_threshold.h5'

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        assert_allclose(self.func(**params), output)