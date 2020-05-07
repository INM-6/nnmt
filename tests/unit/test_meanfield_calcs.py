import pytest

from checks import (check_pos_params_neg_raise_exception,
                    check_correct_output,
                    check_V_0_larger_V_th_raise_exception)

from lif_meanfield_tools.meanfield_calcs import (firing_rates,
                                                 mean,
                                                 standard_deviation,
                                                 transfer_function,
                                                 delay_dist_matrix,
                                                 sensitivity_measure,
                                                 power_spectra,
                                                 eigen_spectra,)


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


class Test_delay_dist_matrix:

    # define tested function
    func = staticmethod(delay_dist_matrix)

    def test_delay_dist_matrix_single_is_called(self, mocker, std_params):
        mocked_tf = mocker.patch('lif_meanfield_tools.meanfield_calcs.'
                                 'delay_dist_matrix_single')
        self.func(**std_params)
        mocked_tf.assert_called_once()


class Test_delay_dist_matrix_single:

    # define tested function
    func = staticmethod(delay_dist_matrix)
    output_key = 'delay_dist_all'

    def test_correct_output_dist_none(self, output_test_fixtures):
        delay_dist = 'none'
        params = output_test_fixtures.pop('params')
        params['delay_dist'] = delay_dist
        output = output_test_fixtures.pop('output')[0]
        check_correct_output(self.func, params, output)

    def test_correct_output_dist_truncated_gaussian(self,
                                                    output_test_fixtures):
        delay_dist = 'truncated_gaussian'
        params = output_test_fixtures.pop('params')
        params['delay_dist'] = delay_dist
        output = output_test_fixtures.pop('output')[1]
        check_correct_output(self.func, params, output)

    def test_correct_output_dist_gaussian(self, output_test_fixtures):
        delay_dist = 'gaussian'
        params = output_test_fixtures.pop('params')
        params['delay_dist'] = delay_dist
        output = output_test_fixtures.pop('output')[2]
        check_correct_output(self.func, params, output)


class Test_sensitivity_measure:

    # define tested function
    func = staticmethod(sensitivity_measure)
    output_key = 'sensitivity_measure'

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params,
                                             pos_keys)

    def test_warning_is_given_if_k_is_critical(self, std_params):
        std_params['tau_s'] = 0.5 * std_params['tau_m']
        with pytest.warns(UserWarning):
            self.func(**std_params)

    def test_exception_is_raised_if_k_is_too_large(self, std_params):
        std_params['tau_s'] = 2 * std_params['tau_m']
        with pytest.raises(ValueError):
            self.func(**std_params)

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        output = output_test_fixtures.pop('output')[0]
        check_correct_output(self.func, params, output)


class Test_power_spectra:

    # define tested function
    func = staticmethod(power_spectra)
    output_key = 'power_spectra'

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params,
                                             pos_keys)

    def test_warning_is_given_if_k_is_critical(self, std_params):
        std_params['tau_s'] = 0.5 * std_params['tau_m']
        with pytest.warns(UserWarning):
            self.func(**std_params)

    def test_exception_is_raised_if_k_is_too_large(self, std_params):
        std_params['tau_s'] = 2 * std_params['tau_m']
        with pytest.raises(ValueError):
            self.func(**std_params)

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        output = output_test_fixtures.pop('output')
        check_correct_output(self.func, params, output)


class Test_eigen_spectra_eval:

    # define tested function
    func = staticmethod(eigen_spectra)
    output_key = 'eigenvalue_spectra'

    def test_pos_params_neg_raise_exception(self, std_params_eval_spectra,
                                            pos_keys):
        check_pos_params_neg_raise_exception(self.func,
                                             std_params_eval_spectra,
                                             pos_keys)

    def test_warning_is_given_if_k_is_critical(self, std_params_eval_spectra):
        std_params_eval_spectra['tau_s'] = 0.5*std_params_eval_spectra['tau_m']
        with pytest.warns(UserWarning):
            self.func(**std_params_eval_spectra)

    def test_exception_is_raised_if_k_is_too_large(self,
                                                   std_params_eval_spectra):
        std_params_eval_spectra['tau_s'] = 2 * std_params_eval_spectra['tau_m']
        with pytest.raises(ValueError):
            self.func(**std_params_eval_spectra)

    def test_correct_output_eigvals_MH(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        params['quantity'] = 'eigvals'
        params['matrix'] = 'MH'
        output = output_test_fixtures.pop('output')[0]
        check_correct_output(self.func, params, output)

    def test_correct_output_eigvals_prop(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        params['quantity'] = 'eigvals'
        params['matrix'] = 'prop'
        output = output_test_fixtures.pop('output')[1]
        check_correct_output(self.func, params, output)

    def test_correct_output_eigvals_prop_inv(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        params['quantity'] = 'eigvals'
        params['matrix'] = 'prop_inv'
        output = output_test_fixtures.pop('output')[2]
        check_correct_output(self.func, params, output)


class Test_eigen_spectra_reigvecs:

    # define tested function
    func = staticmethod(eigen_spectra)
    output_key = 'r_eigenvec_spectra'

    def test_correct_output_reigvecs_MH(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        params['quantity'] = 'reigvecs'
        params['matrix'] = 'MH'
        output = output_test_fixtures.pop('output')[0]
        check_correct_output(self.func, params, output)

    def test_correct_output_reigvecs_prop(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        params['quantity'] = 'reigvecs'
        params['matrix'] = 'prop'
        output = output_test_fixtures.pop('output')[1]
        check_correct_output(self.func, params, output)

    def test_correct_output_reigvecs_prop_inv(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        params['quantity'] = 'reigvecs'
        params['matrix'] = 'prop_inv'
        output = output_test_fixtures.pop('output')[2]
        check_correct_output(self.func, params, output)


class Test_eigen_spectra_leigvecs:

    # define tested function
    func = staticmethod(eigen_spectra)
    output_key = 'l_eigenvec_spectra'

    def test_correct_output_leigvecs_MH(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        params['quantity'] = 'leigvecs'
        params['matrix'] = 'MH'
        output = output_test_fixtures.pop('output')[0]
        check_correct_output(self.func, params, output)

    def test_correct_output_leigvecs_prop(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        params['quantity'] = 'leigvecs'
        params['matrix'] = 'prop'
        output = output_test_fixtures.pop('output')[1]
        check_correct_output(self.func, params, output)

    def test_correct_output_leigvecs_prop_inv(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        params['quantity'] = 'leigvecs'
        params['matrix'] = 'prop_inv'
        output = output_test_fixtures.pop('output')[2]
        check_correct_output(self.func, params, output)


# here tests for Senk start

class Test_additional_rates_for_fixed_input:
    pass


class Test_fit_transfer_function:
    pass


class Test_scan_fit_transfer_function_mean_std_input:
    pass


class Test_effective_coupling_strength:
    pass


class Test_linear_interpolation_alpha:
    pass


class Test_eigenvals_branches_rate:
    pass


class Test_lambda_of_alpha_integral:
    pass


class Test_d_lambda_d_alpha:
    pass


class Test_xi_eff_s:
    pass


class Test_xi_eff_r:
    pass


class Test_d_xi_eff_s_d_lambda:
    pass


class Test_d_xi_eff_r_d_lambda:
    pass


class Test_solve_chareq_numerically_alpha:
    pass


class Test_xi_of_k:
    pass


class Test_solve_chareq_rate_boxcar:
    pass
