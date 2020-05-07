import numpy as np
import pytest


def check_pos_params_neg_raise_exception(func, params, pos_key):
    """Test whether exception is raised if always pos param gets negative."""
    params[pos_key] *= -1
    with pytest.raises(ValueError):
        func(**params)


def check_correct_output(func, params, output, updates=None):
    if updates:
        params = params.copy()
        params.update(updates)
    np.testing.assert_array_equal(func(**params), output)


def check_V_0_larger_V_th_raise_exception(func, params):
    params['V_0_rel'] = 1.1 * params['V_th_rel']
    with pytest.raises(ValueError):
        func(**params)


def check_warning_is_given_if_k_is_critical(func, params):
    params['tau_s'] = 0.5 * params['tau_m']
    with pytest.warns(UserWarning):
        func(**params)


def check_exception_is_raised_if_k_is_too_large(func, params):
    params['tau_s'] = 2 * params['tau_m']
    with pytest.raises(ValueError):
        func(**params)
