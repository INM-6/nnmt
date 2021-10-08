import pytest
import numpy as np

from ...checks import (
    assert_array_equal,
    assert_units_equal,
    assert_allclose,
    )
import nnmt

ureg = nnmt.ureg


class Test_calculation_of_dependent_network_params:
    """
    Depends strongly on network_params_microcircuit.yaml in
    tests/fixtures/unit/config/.
    """

    def test_dimension(self, microcircuit):
        assert microcircuit.network_params['dimension'] == 8

    def test_V0_rel(self, microcircuit):
        assert microcircuit.network_params['V_0_rel'] == 0

    def test_V_th_rel(self, microcircuit):
        assert microcircuit.network_params['V_th_rel'] == 0.015

    def test_j(self, microcircuit):
        assert microcircuit.network_params['j'] == 0.0001756

    def test_W(self, microcircuit):
        W = np.array([[87.8, -351.2, 87.8, -351.2, 87.8, -351.2, 87.8, -351.2]
                      for i in range(microcircuit.network_params['dimension'])]
                     ) * 1e-12

        W[0][2] *= 2
        assert_array_equal(microcircuit.network_params['W'], W)
        assert_units_equal(microcircuit.network_params['W'], W)

    def test_J(self, microcircuit):
        J = np.array([[0.1756, -0.7024, 0.1756, -0.7024, 0.1756, -0.7024,
                       0.1756, -0.7024]
                      for i in range(microcircuit.network_params['dimension'])]
                     ) * 1e-3
        J[0][2] *= 2
        assert_array_equal(microcircuit.network_params['J'], J)
        assert_units_equal(microcircuit.network_params['J'], J)

    def test_Delay(self, microcircuit):
        Delay = np.array([[1.5, 0.75, 1.5, 0.75, 1.5, 0.75, 1.5, 0.75]
                          for i in range(
                              microcircuit.network_params['dimension'])
                          ]) * 1e-3
        assert_array_equal(microcircuit.network_params['Delay'], Delay)
        assert_units_equal(microcircuit.network_params['Delay'], Delay)

    def test_Delay_sd(self, microcircuit):
        Delay_sd = np.array([[0.75, 0.375, 0.75, 0.375, 0.75, 0.375, 0.75,
                              0.375]
                             for i in range(
                                 microcircuit.network_params['dimension'])
                             ]) * 1e-3
        assert_array_equal(microcircuit.network_params['Delay_sd'], Delay_sd)
        assert_units_equal(microcircuit.network_params['Delay_sd'], Delay_sd)


class Test_calculation_of_dependent_analysis_params:
    """Depends strongly on analysis_params_test.yaml in tests/fixtures."""

    def test_omegas(self, microcircuit):
        omegas = [6.28318531e-01,
                  1.89123878e+02,
                  3.77619437e+02,
                  5.66114996e+02,
                  7.54610555e+02,
                  9.43106115e+02,
                  1.13160167e+03,
                  1.32009723e+03,
                  1.50859279e+03,
                  1.69708835e+03]
        assert_allclose(microcircuit.analysis_params['omegas'],
                        omegas, 1e-5)

    def test_k_wavenumbers(self, microcircuit):
        k_wavenumbers = np.array([1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
                                 ) * 1000
        assert_array_equal(microcircuit.analysis_params['k_wavenumbers'],
                           k_wavenumbers)
        assert_units_equal(microcircuit.analysis_params['k_wavenumbers'],
                           k_wavenumbers)

    def test_k_not_available_runs_through_initialization(self):
        run_through = True
        try:
            nnmt.models.Microcircuit(
                network_params=('tests/fixtures/unit/config/'
                                'network_params_microcircuit.yaml'),
                analysis_params={'df': 0.1, 'f_min': 0, 'f_max': 10}
            )
        except:
            run_through = False
        assert run_through
