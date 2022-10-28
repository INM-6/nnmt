import pytest
import numpy as np
from numpy.testing import (
    assert_allclose,
    )

import nnmt


class Test_delay_dist_matrix:

    func = staticmethod(nnmt.network_properties._delay_dist_matrix)
    ids = ['none', 'truncated_gaussian', 'gaussian']
    output_keys = ['delay_dist_{}'.format(id) for id in ids]

    @pytest.mark.parametrize('key', [0, 1, 2], ids=ids)
    def test_correct_output_dist(self, output_test_fixtures, key):
        delay_dist = self.ids[key]
        params = output_test_fixtures['params']
        nnmt.utils._to_si_units(params)
        nnmt.utils._strip_units(params)
        params['delay_dist'] = delay_dist
        output = output_test_fixtures['output'][key]
        output = output.magnitude
        assert_allclose(self.func(**params), output)


class Test_lognormal_characteristic_function:

    @pytest.mark.parametrize('mu, sigma, w', [[1, 1, 1 * np.pi],
                                              [10, 1, 1 * np.pi],
                                              [0.1, 1, 1 * np.pi],
                                              [1, 10, 1 * np.pi],
                                              [1, 0.1, 1 * np.pi],
                                              [1, 1, 10 * np.pi],
                                              [1, 1, 0.1 * np.pi],
                                              [0.0015, 0.0015, 1],
                                              [0.0015, 0.0015, 10],
                                              [0.0015, 0.0015, 100],
                                              [0.0015, 0.0015, 1000],
                                              [0.00075, 0.001, 1],
                                              [0.00075, 0.001, 10],
                                              [0.00075, 0.001, 100],
                                              [0.00075, 0.001, 1000],
                                             ])
    def test_integration_procedure(self, mu, sigma, w):
        N = 10000000
        rtol=0.01
        X = np.random.lognormal(mu, sigma, size=N)
        cf_w_est = np.exp(1j * w * X).mean()
        cf_w_nnmt = (
            nnmt.network_properties._lognormal_characteristic_function(
                w, mu, sigma))
        np.testing.assert_allclose([cf_w_nnmt], [cf_w_est], rtol=rtol)

