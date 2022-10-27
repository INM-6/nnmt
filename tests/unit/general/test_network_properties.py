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

    def test_integration_procedure(self):
        mu = 1
        sigma = 1
        w = np.pi
        N = 10000000
        rtol=0.01
        X = np.random.lognormal(mu, sigma, size=N)
        cf_w_est = np.exp(1j * w * X).mean()
        cf_w_nnmt = (
            nnmt.network_properties._lognormal_characteristic_function(
                w, mu, sigma))
        np.testing.assert_allclose([cf_w_nnmt], [cf_w_est], rtol=rtol)

