import pytest
import re
import inspect
# note that numpy asserts are wrapped with pint_wrap a few lines down
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_allclose)

import lif_meanfield_tools as lmt
ureg = lmt.ureg


def pint_wrap(func):
    """
    Pint wrapper for numpy assert statements.
    
    Numpy asserts given a quantity give an UnstrippedWarning. This wrapper
    makes them handle quantities correctly.
    
    Note that only magnitudes are passed to the wrapped function.
    """
    def decorator_wrap(*args, **kwargs):
        signature = inspect.signature(func)
        # convert tuple to list, such that item assignment is possible
        args = list(args)
        # put all arguments into args
        for key in signature.parameters:
            if key in kwargs:
                args.append(kwargs.pop(key))
        # strip quantity of first two arguments
        for i in range(2):
            try:
                args[i] = args[i].magnitude
            except AttributeError:
                pass
        return func(*args, **kwargs)
    return decorator_wrap


assert_array_equal = pint_wrap(assert_array_equal)
assert_array_almost_equal = pint_wrap(assert_array_almost_equal)
assert_allclose = pint_wrap(assert_allclose)


def assert_quantity_array_equal(qarray1, qarray2):
    assert_array_equal(qarray1, qarray2)
    assert_units_equal(qarray1, qarray2)


def assert_units_equal(var_1, var_2):
    """Checks whether unit of var_1 and of var_2 conincide."""
    try:
        assert var_1.units == var_2.units
    except AttributeError:
        pass
    

def assert_dimensionality_equal(var_1, var_2):
    """Checks whether unit of var_1 and of var_2 have same dimensionality."""
    try:
        assert var_1.dimensionality == var_2.dimensionality
    except AttributeError:
        pass
    

def check_pos_params_neg_raise_exception(func, params, pos_key):
    """Test whether exception is raised if always pos param gets negative."""
    params[pos_key] *= -1
    with pytest.raises(ValueError):
        func(**params)


def check_correct_output(func, params, output, updates=None):
    if updates:
        params = params.copy()
        params.update(updates)
    result = func(**params)
    assert_array_equal(result, output)
    assert_units_equal(result, output)


def check_correct_output_for_several_mus_and_sigmas(func, params, outputs):
    params = params.copy()
    mus = params.pop('mu')
    sigmas = params.pop('sigma')
    for mu, sigma, output in zip(mus, sigmas, outputs):
        params['mu'] = mu
        params['sigma'] = sigma
        result = func(**params)
        if isinstance(result, ureg.Quantity):
            result.ito(output.units)
        assert_array_equal(output, result)
        assert_units_equal(output, result)


def check_almost_correct_output_for_several_mus_and_sigmas(func, alt_func,
                                                           params,
                                                           rtol):
    mus = params.pop('mu')
    sigmas = params.pop('sigma')
    for mu, sigma in zip(mus, sigmas):
        params['mu'] = mu
        params['sigma'] = sigma
        result = func(**params)
        expected = alt_func(**params)
        if isinstance(result, ureg.Quantity):
            expected.ito(result.units)
        assert_allclose(expected, result, rtol)
        assert_units_equal(expected, result)


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
            if isinstance(dict1[key], dict):
                check_quantity_dicts_are_equal(dict1[key], dict2[key])
            else:
                assert_array_equal(dict1[key], dict2[key])
                assert_units_equal(dict1[key], dict2[key])


def check_dict_contains_no_quantity(d):
    for key, value in d.items():
        if isinstance(value, dict):
            check_dict_contains_no_quantity(value)
        else:
            assert not isinstance(value, ureg.Quantity)


def check_dict_contains_no_val_unit_dict(d):
    for key, value in d.items():
        if (key == 'val') or (key == 'unit'):
            assert False
        elif isinstance(value, dict):
            check_dict_contains_no_val_unit_dict(value)
