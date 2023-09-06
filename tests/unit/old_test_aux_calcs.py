import pytest
import numpy as np

from ..checks import (
    check_pos_params_neg_raise_exception,
    check_correct_output_for_several_mus_and_sigmas,
    check_V_0_larger_V_th_raise_exception,
    check_warning_is_given_if_k_is_critical,
    check_exception_is_raised_if_k_is_too_large,
    )

import nnmt
from nnmt.aux_calcs import (
    d_nu_d_nu_in_fb,
    p_hat_boxcar,
    )

ureg = nnmt.ureg

fixture_path = 'fixtures/unit/data/'


def strip_units_from_quantity_dict(d):
    """ Returns dictionary only containing magnitudes. """
    return {key: value.magnitude for key, value in d.items()}


class Test_d_nu_d_nu_in_fb:

    func = staticmethod(d_nu_d_nu_in_fb)
    output_keys = ['d_nu_d_nu_in_fb_mu', 'd_nu_d_nu_in_fb_sigma',
                   'd_nu_d_nu_in_fb_all']

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params, pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_params)

    def test_warning_is_given_if_k_is_critical(
            self, std_params_single_population):
        check_warning_is_given_if_k_is_critical(
            self.func, std_params_single_population)

    def test_exception_is_raised_if_k_is_too_large(self, std_params):
        check_exception_is_raised_if_k_is_too_large(self.func, std_params)

    def test_zero_sigma_raises_error(self, std_params):
        std_params['sigma'] = 0
        with pytest.raises(ZeroDivisionError):
            self.func(**std_params)

    @pytest.mark.parametrize('contribution, num', [('mu', 0), ('sigma', 1),
                                                   ('all', 2)])
    def test_correct_output(self, output_test_fixtures, contribution, num):
        params = output_test_fixtures['params']
        outputs = output_test_fixtures['output']
        params['contributions'] = contribution
        output = outputs[num]
        check_correct_output_for_several_mus_and_sigmas(self.func, params,
                                                        output)


class Test_p_hat_boxcar:

    func = staticmethod(p_hat_boxcar)

    def test_correct_output(self):
        fixtures = np.load(fixture_path + 'p_hat_boxcar.npz')
        ks = fixtures['ks']
        widths = fixtures['widths']
        outputs = fixtures['outputs']
        for z, x, output in zip(ks, widths, outputs):
            result = self.func(z, x)
            assert result == output
