import pytest

import inspect
import re
import numpy as np

import lif_meanfield_tools as lmt
from lif_meanfield_tools.meanfield_calcs import *

ureg = lmt.ureg

fixtures_input_path = 'tests/unit/fixtures/input/'
fixtures_output_path = 'tests/unit/fixtures/output/'

all_pos_params = ['a', 'nu', 'K', 'tau_m', 'nu_ext', 'K_ext', 
                  'nu_e_ext', 'nu_i_ext']

    
def check_negative_parameters_that_should_be_positive_raise_exception(
        function, params, pos_param):
        params[pos_param] *= -1
        with pytest.raises(ValueError):
            function(**params)
            

def check_correct_output(function, params, output, updates=None):
    if updates:
        params = params.copy()
        params.update(updates)
    assert function(**params) == output


@pytest.fixture
def standard_params(self):
    return dict(nu = 1*ureg.Hz,
                K = 1,
                J = 1*ureg.mV,
                j = 1*ureg.mV,
                tau_m = 18*ureg.s,
                nu_ext = 1*ureg.Hz,
                K_ext = 1,
                g = 1,
                nu_e_ext = 1*ureg.Hz,
                nu_i_ext = 1*ureg.Hz)
    
    
def get_standard_params(function):
    """ Creates parameter fixture for given function. """
    @pytest.fixture
    def standard_params(standard_params):
        """ Returns standard params needed for respective function. """
        _params = {k:v for k,v in standard_params 
                   if k in inspect.signature(function).parameters}
        return _params
    return standard_params


def get_pos_params(function, all_pos_params):
    """ Creates a fixture for all positive params of given function. """
    # all params in all_pos_params and required as function argument
    pos_params = [param for param in all_pos_params 
                  if param in inspect.signature(function).parameters]
    
    @pytest.fixture(params=pos_params)
    def pos_params(request):
        """ Parametrizes positive parameters. """
        return request.param

    return pos_params



class TestMean():

    # define tested function
    function = mean
    
    # define fixtures
    standard_params = get_standard_params(function)
    pos_params = get_pos_params(function, all_pos_params)
    
    # @pytest.mark.parametrize('pos_param', pos_params)
    def test_negative_parameters_that_should_be_positive_raise_exception(
            self, standard_params, pos_params):
        check_negative_parameters_that_should_be_positive_raise_exception(
            self.function, standard_params, pos_params)

    # @pytest.mark.parametrize('updates, output', 
    #                          [(dict(nu=10*ureg.Hz), 198*ureg.mV)])
    # def test_correct_output_in_noise_driven_regime(self, params, output, 
    #                                                updates):
    #     check_correct_output(self.function, params, output, updates)
    # 
    # @pytest.mark.parametrize('updates, output', 
    #                          [(dict(nu=10*ureg.Hz), 3*ureg.Hz)])
    # def test_correct_output_in_mean_driven_regime(self, params, output, 
    #                                               updates):
    #     check_correct_output(self.function, params, output, updates)
    # 
    # @pytest.mark.parametrize('updates, output', 
    #                          [(dict(nu=10*ureg.Hz), 3*ureg.Hz)])
    # def test_correct_output_in_negative_firing_rate_regime(self, params, output, 
    #                                                        updates):
    #     check_correct_output(self.function, params, output, updates)
    # 
    
    
# 
# @pytest.fixture
# def parameters():
#     return dict(nu = 1*ureg.Hz,
#                           K = 1,
#                           J = 1*ureg.mV,
#                           j = 1*ureg.mV,
#                           tau_m = 18*ureg.s,
#                           nu_ext = 1*ureg.Hz,
#                           K_ext = 1,
#                           g = 1,
#                           nu_e_ext = 1*ureg.Hz,
#                           nu_i_ext = 1*ureg.Hz,)
# 
# @pytest.fixture
# def parameters_2(parameters):
#     return parameters.update
# def pama
# 
# @pytest.fixture
# def correct_output_mean():
#     return 36 * ureg.mV
# 
# def test_correct_output(parameters, correct_output_mean):
#     result = mean(**parameters) 
#     assert result == correct_output_mean
# 

# 
# 
# import unittest
# from unittest import mock
# from unittest.mock import patch
# 
# import numpy as np
# 
# import lif_meanfield_tools as lmt
# from lif_meanfield_tools.meanfield_calcs import *
# 
# ureg = lmt.ureg
# 
# fixtures_input_path = 'tests/unit/fixtures/input/'
# fixtures_output_path = 'tests/unit/fixtures/output/'
# 
# 
# class GeneralTester(unittest.TestCase):
# 
#     def __init__(self, function, params, positive_params, precision=10):
#         self.function = function
#         self.params = params
#         self.positive_params = positive_params
#         self.precision = precision
# 
#     def check_negative_parameters_that_should_be_positive_raise_exception(self):
#         for param in self.positive_params:
#             temp_params = self.params
#             temp_params[param] *= -1
#             with self.assertRaises(ValueError):
#                 self.function(**temp_params)

    # def test_correct_output(self):
    #     for param, expected_output in zip(params, expected_outputs):
    #         temp_param = self.parameters.copy()
    #         temp_param.update(param)                                
    #         output = self.function(**temp_param)
    #         self.assertAlmostEqual(expected_output, output, self.precision)

# 
# class Test_mean(unittest.TestCase):
#     # 
#     # _general_tests = ['test_negative_parameters_that_should_be_positive_raise_exception', 
#     #                   'test_correct_output']
#     # 
#     def __init__(self):
#         self.params = dict(nu = 1, 
#                            K = 1, 
#                            J = 1,
#                            j = 1, 
#                            tau_m = 1, 
#                            nu_ext = 1, 
#                            K_ext = 1, 
#                            g = 1, 
#                            nu_e_ext = 1, 
#                            nu_i_ext = 1)
#         self.positive_params = ['nu', 'K', 'tau_m', 'nu_ext', 'K_ext', 
#                                 'nu_e_ext', 'nu_i_ext']
#         self._tester = GeneralTester(mean, self.params, self.positive_params)
#     # 
#     # def __getattr__(self, attribute):
#     #     if attribute in self._general_tests:
#     #         return getattr(self._tester, attribute)
# 
#     def test_negative_parameters_that_should_be_positive_raise_exception(self):
#         self._tester.check_negative_parameters_that_should_be_positive_raise_exception()
# 
# 
# class Test_standard_deviation(unittest.TestCase):
# 
#     @property
#     def positive_params(self):
#         return ['nu', 'K', 'tau_m', 'nu_ext', 'K_ext', 'nu_e_ext', 'nu_i_ext']
# 
#     def test_negative_parameters_that_should_be_positive_raise_exception(self):
#         pass
# 
#     def test_correct_output(self):
#         pass 
# 
# 
# class Test_firing_rates(unittest.TestCase):
# 
#     @property
#     def positive_params(self):
#         return ['dimension', 'tau_m', 'tau_s', 'tau_r', 'K', 'nu_ext', 'K_ext', 'nu_e_ext', 'nu_i_ext']
# 
#     def test_negative_parameters_that_should_be_positive_raise_exception(self):
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
#     def test_correct_output_in_negative_firing_rate_regime(self):
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
#     def test_correct_output_in_negative_firing_rate_regime(self):
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
#     def test_correct_output_in_negative_firing_rate_regime(self):
#         pass
# 
#     def test_for_zero_frequency_d_nu_d_mu_fb433_is_called(self):
#         pass
# 
# 
# class Test_transfer_function(unittest.TestCase):
# 
#     def test_correct_function_is_called(self):
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
