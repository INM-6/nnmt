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
                                              [0.1, 1, 1 * np.pi],
                                              [1, 0.1, 1 * np.pi],
                                              [1, 1, 0.1 * np.pi],
                                              [-7.706262975199909, 1.0107676525947895, 1],
                                              [-7.706262975199909, 1.0107676525947895, 10],
                                              [-7.706262975199909, 1.0107676525947895, 100],
                                              [-6.848863761153945, 0.8325546111576977, 1],
                                              [-6.848863761153945, 0.8325546111576977, 10],
                                              [-6.848863761153945, 0.8325546111576977, 100],
                                             ])
    def test_integration_procedure(self, mu, sigma, w):
        # This tests works by drawing a fixed number of samples from a
        # lognormal distribution and using these to compute the characteristic
        # function of the lognormal distribution numerically. The resulting
        # value is then compared to the result of the function to be tested.

        # number of samples
        N = 10000000
        # relative tolerance of results computed via the two procedures
        rtol=0.012

        # draw samples
        X = np.random.lognormal(mu, sigma, size=N)
        # compute characteristic function numerically
        cf_w_est = np.exp(1j * w * X).mean()

        # compute characteristic function with nnmt function
        cf_w_nnmt = (
            nnmt.network_properties._lognormal_characteristic_function(
                w, mu, sigma))

        # compare the results
        np.testing.assert_allclose([cf_w_nnmt], [cf_w_est], rtol=rtol)


    @pytest.mark.xfail
    @pytest.mark.parametrize('mu, sigma, w', [[10, 1, 1 * np.pi],
                                              [1, 10, 1 * np.pi],
                                              [1, 1, 10 * np.pi],
                                             ])
    def test_integration_procedure_with_failing_values(self, mu, sigma, w):
        # This is the same test as above, but the values used are the one where
        # the convergence with the method from Beaulieu 2008 seems to fail. We
        # use this test to document the failing values. For the respective
        # ranges another complementary method could be implemented.

        # number of samples
        N = 10000000
        # relative tolerance of results computed via the two procedures
        rtol=0.01

        # draw samples
        X = np.random.lognormal(mu, sigma, size=N)
        # compute characteristic function numerically
        cf_w_est = np.exp(1j * w * X).mean()

        # compute characteristic function with nnmt function
        cf_w_nnmt = (
            nnmt.network_properties._lognormal_characteristic_function(
                w, mu, sigma))

        # compare the results
        np.testing.assert_allclose([cf_w_nnmt], [cf_w_est], rtol=rtol)


    @pytest.mark.parametrize('mu, sigma', [[0.00075, 0.001],
                                           [0.0015, 0.0015],
                                           [10, 1]])
    def test_calc_of_underlying_gaussian_params(self, mu, sigma):
        N = 10000000
        rtol = 0.01
        mu_gauss = nnmt.network_properties._mu_underlying_gaussian(mu, sigma)
        sigma_gauss = nnmt.network_properties._sigma_underlying_gaussian(
            mu, sigma)
        X = np.random.lognormal(mu_gauss, sigma_gauss, N)
        mu_est = X.mean()
        sigma_est = X.std()
        np.testing.assert_allclose([mu_est, sigma_est], [mu, sigma], rtol=rtol)


    def test_only_integrates_as_often_as_required(self, mocker):
        D = np.array([[0.0015, 0.00075],
                      [0.0015, 0.00075]])
        D_sd = np.array([[0.0015, 0.001],
                         [0.0015, 0.001]])
        omegas = np.array([1, 2, 3]) * 2 * np.pi


        mock_integration = mocker.patch(
            'nnmt.network_properties._lognormal_characteristic_function',
            return_value=1
            )

        nnmt.network_properties._delay_dist_matrix(
            D, D_sd, 'lognormal', omegas)

        assert mock_integration.call_count == 6


    def test_returns_correct_results(self):
        D = np.array([[0.0015, 0.00075],
                      [0.0015, 0.00075]])
        D_sd = np.array([[0.0015, 0.001],
                         [0.0015, 0.001]])
        omegas = np.array([1, 2, 3])

        output = nnmt.network_properties._delay_dist_matrix(
            D, D_sd, 'lognormal', omegas)

        mu = nnmt.network_properties._mu_underlying_gaussian(0.00075, 0.001)
        sigma = nnmt.network_properties._sigma_underlying_gaussian(
            0.00075, 0.001)
        output_exp_001 = (
            nnmt.network_properties._lognormal_characteristic_function(
                1, mu, sigma))
        mu = nnmt.network_properties._mu_underlying_gaussian(0.0015, 0.0015)
        sigma = nnmt.network_properties._sigma_underlying_gaussian(
            0.0015, 0.0015)
        output_exp_210 = (
            nnmt.network_properties._lognormal_characteristic_function(
                3, mu, sigma))
        output = nnmt.network_properties._delay_dist_matrix(
            D, D_sd, 'lognormal', omegas)

        assert output[0, 0, 1] == output_exp_001
        assert output[2, 1, 0] == output_exp_210