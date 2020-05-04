import pytest

from checks import (check_pos_params_neg_raise_exception,
                    check_correct_output,
                    check_V_0_larger_V_th_raise_exception)

from ..utils import get_required_params

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

    @pytest.fixture
    def std_params(self, all_std_params):
        """Returns set of standard params needed for all tests."""
        return get_required_params(self.func, all_std_params)

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

    @pytest.fixture
    def std_params(self, all_std_params):
        """Returns set of standard params needed for all tests."""
        return get_required_params(self.func, all_std_params)

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

    @pytest.fixture
    def std_params(self, all_std_params):
        """Returns set of standard params needed for all tests."""
        return get_required_params(self.func, all_std_params)

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params, pos_keys)

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        output = output_test_fixtures.pop('output')
        check_correct_output(self.func, params, output)


class Test_transfer_function:

    # define tested functiosn
    func = staticmethod(transfer_function)

    @pytest.fixture
    def std_params(self, all_std_params):
        """Returns set of standard params needed for all tests."""
        params = get_required_params(self.func, all_std_params)
        params['mu'] = [1*ureg.mV, 2*ureg.mV]
        params['sigma'] = [1*ureg.mV, 2*ureg.mV]
        return params

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params, pos_keys)

    def test_shift_method_is_called(self, mocker, std_params):
        mocked_tf = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                                 'transfer_function_1p_shift')
        std_params['method'] = 'shift'
        self.func(**std_params)
        mocked_tf.assert_called_once()

    def test_taylor_method_is_called(self, mocker, std_params):
        mocked_tf = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                                 'transfer_function_1p_taylor')
        std_params['method'] = 'taylor'
        self.func(**std_params)
        mocked_tf.assert_called_once()


class Test_transfer_function_1p_shift():

    # define tested functiosn
    func = staticmethod(transfer_function_1p_shift)
    output_key = 'std_input'

    @pytest.fixture
    def std_params(self, all_std_params):
        """Returns set of standard params needed for all tests."""
        return get_required_params(self.func, all_std_params)

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params, pos_keys)

    def test_warning_is_given_if_k_is_critical(self, std_params):
        std_params['tau_s'] = 0.5 * std_params['tau_m']
        with pytest.warns(UserWarning):
            self.func(**std_params)

    def test_exception_is_raised_if_k_is_too_large(self, std_params):
        std_params['tau_s'] = 2 * std_params['tau_m']
        with pytest.raises(ValueError):
            self.func(**std_params)

    # def test_correct_output(self, params_all_regimes):
    #     output = params_all_regimes['tf_shift']
    #     required_params = get_required_params(self.func, params_all_regimes)
    #     check_correct_output(self.func, required_params, output)


class Test_transfer_function_1p_taylor():
    pass

#
# class Test_firing_rates(unittest.TestCase):
#
#     @property
#     def positive_params(self):
#         return ['dimension', 'tau_m', 'tau_s', 'tau_r', 'K', 'nu_ext', 'K_ext', 'nu_e_ext', 'nu_i_ext']
#
#     def test_pos_params_neg_raise_exception(self):
#         pass
#
#     def test_V_0_larger_V_th_raise_exception(self):
#         pass
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
#
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
