import pytest
import numpy as np

from numpy.testing import (
    assert_allclose
    )

from nnmt import linear_stability

class Test_chareq_lambertw_constant_delay:

    func = staticmethod(linear_stability._solve_chareq_lambertw_constant_delay)
    params = {
        'branch_nr': 0,
        'tau': 0.003,
        'delay': 0.002,
        'connectivity': np.array([[2., -10.],
                                  [2., -10]])}

    def test_lambertw_solution(self):
        """ Test if Eq. 3 of Senk et al. (2020) is solved. """
        lam = self.func(**self.params)
        lhs = ((1. + self.params['tau'] * lam)
                * np.exp(lam * self.params['delay']))

        cs = np.linalg.eigvals(self.params['connectivity'])
        rhs = cs[np.argmax(np.abs(cs))]

        assert_allclose(lhs, rhs)

    def test_vectorized_parameters(self):
        params = dict(self.params)
        params['tau'] = [0.003, 0.003]

        assert_allclose(self.func(**self.params), self.func(**params))

    def test_nonequal_values_raise_error(self):
        params = dict(self.params)
        params['tau'] = [0.003, 0.004]
 
        with pytest.raises(AssertionError):
            self.func(**params)




