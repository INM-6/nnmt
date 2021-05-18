import pytest
from numpy.testing import (
    assert_allclose,
    )

import lif_meanfield_tools as lmt


class Test_delay_dist_matrix:

    func = staticmethod(lmt.network_properties._delay_dist_matrix)
    ids = ['none', 'truncated_gaussian', 'gaussian']
    output_keys = ['delay_dist_{}'.format(id) for id in ids]

    @pytest.mark.parametrize('key', [0, 1, 2], ids=ids)
    def test_correct_output_dist(self, output_test_fixtures, key):
        delay_dist = self.ids[key]
        params = output_test_fixtures['params']
        lmt.utils._to_si_units(params)
        lmt.utils._strip_units(params)
        params['delay_dist'] = delay_dist
        output = output_test_fixtures['output'][key]
        output = output.magnitude
        assert_allclose(self.func(**params), output)
