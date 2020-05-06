import pytest
import numpy as np

from checks import (check_pos_params_neg_raise_exception,
                    check_correct_output,
                    check_V_0_larger_V_th_raise_exception)

from lif_meanfield_tools import ureg
from lif_meanfield_tools.meanfield_calcs import *


class Test_firing_rates:
    """Tests firing rates function. Probably this is a functional test."""

    # Def function as staticmethod because it is not tightly related to class,
    # but we still want to attach it to the class for later reference. This
    # allows calling function as an 'unbound function', without passing the
    # instance to the function:
    # `self.func()` = `func()` != `func(self)`.
    func = staticmethod(firing_rates)
    output_key = 'firing_rates'

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params, pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_params)

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        output = output_test_fixtures.pop('output')
        check_correct_output(self.func, params, output)


class Test_mean:

    # further explanation see Test_firing_rates
    func = staticmethod(mean)
    output_key = 'mean_input'

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params, pos_keys)

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        output = output_test_fixtures.pop('output')
        check_correct_output(self.func, params, output)


class Test_standard_deviation:

    # further explanation see Test_firing_rates
    func = staticmethod(standard_deviation)
    output_key = 'std_input'

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params, pos_keys)

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        output = output_test_fixtures.pop('output')
        check_correct_output(self.func, params, output)


class Test_transfer_function:

    # define tested function
    func = staticmethod(transfer_function)

    def test_pos_params_neg_raise_exception(self, std_params_tf, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params_tf,
                                             pos_keys)

    def test_shift_method_is_called(self, mocker, std_params_tf):
        mocked_tf = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                                 'transfer_function_1p_shift')
        std_params_tf['method'] = 'shift'
        self.func(**std_params_tf)
        mocked_tf.assert_called_once()

    def test_taylor_method_is_called(self, mocker, std_params_tf):
        mocked_tf = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                                 'transfer_function_1p_taylor')
        std_params_tf['method'] = 'taylor'
        self.func(**std_params_tf)
        mocked_tf.assert_called_once()


class Test_transfer_function_1p_shift():

    # define tested function
    func = staticmethod(transfer_function)
    output_key = 'tf_shift'

    def test_pos_params_neg_raise_exception(self, std_params_tf, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params_tf,
                                             pos_keys)

    def test_warning_is_given_if_k_is_critical(self, std_params_tf):
        std_params_tf['tau_s'] = 0.5 * std_params_tf['tau_m']
        with pytest.warns(UserWarning):
            self.func(**std_params_tf)

    def test_exception_is_raised_if_k_is_too_large(self, std_params_tf):
        std_params_tf['tau_s'] = 2 * std_params_tf['tau_m']
        with pytest.raises(ValueError):
            self.func(**std_params_tf)

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        params['method'] = 'shift'
        output = output_test_fixtures.pop('output')
        check_correct_output(self.func, params, output)


class Test_transfer_function_1p_taylor():

    # define tested function
    func = staticmethod(transfer_function)
    output_key = 'tf_taylor'

    def test_pos_params_neg_raise_exception(self, std_params_tf, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params_tf,
                                             pos_keys)

    def test_warning_is_given_if_k_is_critical(self, std_params_tf):
        std_params_tf['tau_s'] = 0.5 * std_params_tf['tau_m']
        with pytest.warns(UserWarning):
            self.func(**std_params_tf)

    def test_exception_is_raised_if_k_is_too_large(self, std_params_tf):
        std_params_tf['tau_s'] = 2 * std_params_tf['tau_m']
        with pytest.raises(ValueError):
            self.func(**std_params_tf)

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        params['method'] = 'taylor'
        output = output_test_fixtures.pop('output')
        check_correct_output(self.func, params, output)


# class Test_transfer_function_1p_taylor(unittest.TestCase):
#
#     def test_correct_output_in_noise_driven_regime(self):
#         pass
#
#     def test_correct_output_in_mean_driven_regime(self):
#         pass
#
#     def test_correct_output_in_neg_firing_rate_regime(self):
#         pass
#
#     def test_for_zero_frequency_d_nu_d_mu_fb433_is_called(self):
#         pass
#
#
# class Test_transfer_function_1p_shift(unittest.TestCase):
#
#     def test_correct_output_in_noise_driven_regime(self):
#         pass
#
#     def test_correct_output_in_mean_driven_regime(self):
#         pass
#
#     def test_correct_output_in_neg_firing_rate_regime(self):
#         pass
#
#     def test_for_zero_frequency_d_nu_d_mu_fb433_is_called(self):
#         pass
#
#
# class Test_delay_dist_matrix_single(unittest.TestCase):
#
#     def test_correct_output_for_none(self):
#         pass
#
#     def test_correct_output_for_truncated_gaussian(self):
#         pass
#
#     def test_correct_output_for_gaussian(self):
#         pass
#
#
# class Test_effective_connectivity(unittest.TestCase):
#
#     def test_correct_output(self):
#         pass
#
