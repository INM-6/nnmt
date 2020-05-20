import numpy as np
import pytest
import re


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
        
        
def check_file_in_tmpdir(filename, tmp_dir):
    # results file name expression
    exp = re.compile(r'.*{}'.format(filename))
    # file names in tmp dir
    filenames = [str(obj) for obj in tmp_dir.listdir()]
    # file names matching exp
    matches = list(filter(exp.match, filenames))
    # pass test if test file created
    assert any(matches)


def check_quantity_dicts_are_equal(dict1, dict2):
    keys = sorted(dict1.keys())
    for key in keys:
        assert key in dict2
        try:
            assert dict1[key] == dict2[key]
        except ValueError:
            np.testing.assert_array_equal(dict1[key],
                                          dict2[key])
