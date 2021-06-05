import pytest

from ..checks import (
    check_pos_params_neg_raise_exception,
    check_correct_output,
    assert_allclose,
    assert_units_equal,
    check_V_0_larger_V_th_raise_exception,
    )

from lif_meanfield_tools.meanfield_calcs import (
    eigen_spectra,
    additional_rates_for_fixed_input,
    effective_coupling_strength,
    )


class Test_eigen_spectra_eval:

    func = staticmethod(eigen_spectra)
    output_keys = ['eigenvalue_spectra_MH', 'eigenvalue_spectra_prop',
                   'eigenvalue_spectra_prop_inv', 'regime']

    def test_pos_params_neg_raise_exception(self, std_params_eval_spectra,
                                            pos_keys):
        check_pos_params_neg_raise_exception(self.func,
                                             std_params_eval_spectra,
                                             pos_keys)

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
        output = output_test_fixtures.pop('output')
        output = output[2]
        check_correct_output(self.func, params, output)


class Test_eigen_spectra_reigvecs:

    func = staticmethod(eigen_spectra)
    output_keys = ['r_eigenvec_spectra_MH', 'r_eigenvec_spectra_prop',
                   'r_eigenvec_spectra_prop_inv', 'regime']
    
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
        output = output_test_fixtures.pop('output')
        output = output[2]
        check_correct_output(self.func, params, output)


class Test_eigen_spectra_leigvecs:

    func = staticmethod(eigen_spectra)
    output_keys = ['l_eigenvec_spectra_MH', 'l_eigenvec_spectra_prop',
                   'l_eigenvec_spectra_prop_inv', 'regime']

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
        output = output_test_fixtures.pop('output')
        output = output[2]
        check_correct_output(self.func, params, output)


class Test_additional_rates_for_fixed_input:

    func = staticmethod(additional_rates_for_fixed_input)
    output_keys = ['add_nu_e_ext', 'add_nu_i_ext']

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params, pos_keys)

    @pytest.mark.parametrize('key', [0, 1],
                             ids=['nu_e_ext', 'nu_i_ext'])
    def test_correct_output_nu_e_ext(self, output_test_fixtures, key):
        params = output_test_fixtures['params']
        output = output_test_fixtures['output']
        result = self.func(**params)[key]
        assert_allclose(result, output[key])
        assert_units_equal(result, output[key])


class Test_effective_coupling_strength:

    func = staticmethod(effective_coupling_strength)
    output_key = 'effective_coupling_strength'

    def test_pos_params_neg_raise_exception(self, std_params, pos_keys):
        check_pos_params_neg_raise_exception(self.func, std_params, pos_keys)

    def test_V_0_larger_V_th_raise_exception(self, std_params):
        check_V_0_larger_V_th_raise_exception(self.func, std_params)

    def test_correct_output(self, output_test_fixtures):
        params = output_test_fixtures.pop('params')
        output = output_test_fixtures.pop('output')
        check_correct_output(self.func, params, output)
