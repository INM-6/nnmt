import pytest
import numpy as np
from scipy.integrate import quad
from scipy.special import erf, erfcx

from numpy.testing import (
    assert_allclose
    )

from ...checks import (check_pos_params_neg_raise_exception,
                       check_V_0_larger_V_th_raise_exception,
                       )


import nnmt
import nnmt.lif.delta as delta

ureg = nnmt.ureg

fixture_path = 'tests/fixtures/unit/data/'


def integrand(x):
    return np.exp(x**2) * (1 + erf(x))


def real_siegert(tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma):
    """ Siegert formula as given in Fourcaud Brunel 2002 eq. 4.11 """
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    y_th = (V_th_rel - mu) / sigma
    y_r = (V_0_rel - mu) / sigma
    # this brings tau_m and tau_r into the correct vectorized form if they are
    # scalars and doesn't do anything if they are arrays of appropriate size
    tau_m = tau_m + y_th - y_th
    tau_r = tau_r + y_th - y_th

    nu = np.zeros(len(mu))
    for i, (mu, sigma, y_th, y_r, tau_m, tau_r) in enumerate(
            zip(mu, sigma, y_th, y_r, tau_m, tau_r)):
        nu[i] = 1 / (tau_r + np.sqrt(np.pi) * tau_m
                     * quad(integrand, y_r, y_th, epsabs=1e-6)[0])
    return nu


@pytest.fixture
def empty_network():
    """Network object with no parameters."""
    return nnmt.models.Network()


class Test_firing_rates_wrapper:

    func = staticmethod(delta.firing_rates)

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


class Test_firing_rates:

    func = staticmethod(delta._firing_rates_for_given_input)
    fixtures = 'lif_delta_firing_rate.h5'
    rtol = 1e-4

    def test_pos_params_neg_raise_exception(self, std_unitless_params,
                                            pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_unitless_params,
                                             pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_unitless_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_unitless_params)

    def test_gives_similar_results_as_real_siegert_if_siegert_converges(
            self, unit_fixtures):
        params = unit_fixtures.pop('params')

        siegert = real_siegert(**params)
        if not np.any(np.isnan(siegert)):
            assert_allclose(self.func(**params), siegert, rtol=self.rtol)

    def test_correct_output(self, unit_fixtures):
        params = unit_fixtures.pop('params')
        output = unit_fixtures.pop('output')
        assert_allclose(self.func(**params), output)


class Test_siegert_helper:

    def test_erfcx_quadrature_order_detection(self):
        rtol = 1e-12
        a = np.random.uniform(0, 10)
        b = a + np.random.uniform(0, 90)
        params = {'start_order': 1, 'epsrel': rtol, 'maxiter': 20}
        order = delta._get_erfcx_integral_gl_order(y_th=b, y_r=a, **params)
        I_quad = quad(erfcx, a, b, epsabs=0, epsrel=rtol)[0]
        I_gl = delta._erfcx_integral(a, b, order=order)[0]
        err = np.abs(I_gl / I_quad - 1)
        assert err <= rtol

    def test_erfcx_quadrature_analytical_limit(self):
        a = 100  # noise free limit a -> oo
        b = a + np.linspace(1, 100, 100)
        I_ana = np.log(b / a) / np.sqrt(np.pi)  # asymptotic result for a -> oo
        params = {'start_order': 10, 'epsrel': 1e-12, 'maxiter': 20}
        order = delta._get_erfcx_integral_gl_order(y_th=b, y_r=a, **params)
        I_gl = delta._erfcx_integral(a, b, order=order)
        err = np.abs(I_gl / I_ana - 1)
        assert np.all(err <= 1e-4)


class Test_derivative_of_firing_rates_wrt_mean_input:

    func = staticmethod(delta._derivative_of_firing_rates_wrt_mean_input)
    output_key = 'd_nu_d_mu'

    def test_pos_params_neg_raise_exception(self, std_unitless_params,
                                            pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_unitless_params,
                                             pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_unitless_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_unitless_params)

    def test_zero_sigma_raises_error(self, std_params):
        std_params['sigma'] = 0
        with pytest.raises(ZeroDivisionError):
            self.func(**std_params)

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        nnmt.utils._strip_units(params)
        outputs = output_test_fixtures.pop('output')
        assert_allclose(self.func(**params), outputs)
