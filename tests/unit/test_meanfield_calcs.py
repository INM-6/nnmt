import pytest

import inspect
import re
import numpy as np

import lif_meanfield_tools as lmt
from lif_meanfield_tools.meanfield_calcs import *
from lif_meanfield_tools.input_output import load_params

ureg = lmt.ureg

fixture_path = 'tests/unit/fixtures/'

all_standard_params = dict(mu = 1*ureg.mV,
                           sigma = 1*ureg.mV,
                           nu = 1*ureg.Hz,
                           K = 1,
                           J = 1*ureg.mV,
                           j = 1*ureg.mV,
                           V_th_rel = 15*ureg.mV,
                           V_0_rel = 0*ureg.mV,
                           tau_m = 20*ureg.s,
                           tau_s = 1*ureg.s,
                           tau_r = 1*ureg.s,
                           nu_ext = 1*ureg.Hz,
                           K_ext = 1,
                           g = 1,
                           nu_e_ext = 1*ureg.Hz,
                           nu_i_ext = 1*ureg.Hz,
                           dimension = 1,
                           omega = 1*ureg.Hz,
                           omegas = [1*ureg.Hz,])


all_pos_params = ['a', 
                  'nu', 
                  'K', 
                  'tau_m', 
                  'tau_s',
                  'tau_r',
                  'nu_ext', 
                  'K_ext', 
                  'nu_e_ext', 
                  'nu_i_ext',
                  'dimension',
                  'sigma']


# noise driven regime mu < V_th
# parameters taken from microcircuit example
_params_noise_driven_regime = load_params(
    '{}noise_driven_regime.yaml'.format(fixture_path))
_params_noise_driven_regime['mu'] = _params_noise_driven_regime['mean_input']
_params_noise_driven_regime['sigma'] = _params_noise_driven_regime['std_input']
_params_noise_driven_regime['nu'] = _params_noise_driven_regime['firing_rates']

# regime in which negative firing rates occured once
# parameters taken from circuit in which lmt returned negative rates
_params_negative_firing_rate_regime = load_params(
    '{}negative_firing_rate_regime.yaml'.format(fixture_path))
_params_negative_firing_rate_regime['mu'] = _params_negative_firing_rate_regime['mean_input']
_params_negative_firing_rate_regime['sigma'] = _params_negative_firing_rate_regime['std_input']
_params_negative_firing_rate_regime['nu'] = _params_negative_firing_rate_regime[
    'firing_rates']

# # mean driven regime mu > V_th
# # parameters taken from adjusted microcircuit example
# params_mean_driven_regime = load_params(
#                                 'tests/unit/fixtures/mean_driven_regime.yaml')
# params_mean_driven_regime['nu'] = params_mean_driven_regime['firing_rates']

# mus =  [741.89455754, 21.24112709, 35.86521795, 40.69297877, 651.19761921] * ureg.mV
# sigmas = [39.3139564, 6.17632725, 9.79196704, 10.64437979, 37.84928217] * ureg.mV
# 
# tau_m = 20. * ureg.ms
# tau_s = 0.5 * ureg.ms
# tau_r = 0.5 * ureg.ms
# V_th_rel = 20 * ureg.mV
# V_0_rel = 0 * ureg.mV
# 
# tau_ms = np.repeat(tau_m, len(mus))
# tau_ss = np.repeat(tau_s, len(mus))
# tau_rs = np.repeat(tau_r, len(mus))
# V_th_rels = np.repeat(V_th_rel, len(mus))
# V_0_rels = np.repeat(V_0_rel, len(mus))
# 
# _parameters_mean_driven_regime =dict(mu=mus, sigma=sigmas, 
#                                      tau_m=tau_ms, tau_s=tau_ss,
#                                      tau_r=tau_rs, 
#                                      V_th_rel=V_th_rels,
#                                      V_0_rel=V_0_rels)


_ids_all_regimes = ['noise_driven_regime', 
                    'negative_firing_rate_regime',]
                    # 'mean_driven_regime']

_params_all_regimes = [_params_noise_driven_regime,
                       _params_negative_firing_rate_regime,]
                       # params_mean_driven_regime]
                       
_transfer_function_shift = [np.load('{}transfer_function_shift_noise_driven.npy'.format(fixture_path)),
                            np.load('{}transfer_function_shift_negative_firing_rate.npy'.format(fixture_path))]
_transfer_function_taylor = [np.load('{}transfer_function_taylor_noise_driven.npy'.format(fixture_path)),
                            np.load('{}transfer_function_taylor_negative_firing_rate.npy'.format(fixture_path))]

_params_all_regimes = [dict(params, **dict(tf_shift=tf_s, tf_taylor=tf_t))
                       for params, tf_s, tf_t in zip(_params_all_regimes, 
                                                     _transfer_function_shift,
                                                     _transfer_function_taylor)]

@pytest.fixture(params=_params_all_regimes, ids=_ids_all_regimes)
def params_all_regimes(request):
    return request.param

    
def check_negative_parameters_that_should_be_positive_raise_exception(
        function, params, pos_param):
        params[pos_param] *= -1
        with pytest.raises(ValueError):
            function(**params)
            

def check_correct_output(function, params, output, updates=None):
    if updates:
        params = params.copy()
        params.update(updates)
    np.testing.assert_array_equal(function(**params), output)
    
    
def inject_fixture(name, creation_function, *someparams):
    """
    Register dynamically created fixtures such that pytest can find them.
    
    Parameters
    ----------
    name : str
        Name of fixture.
    creation_function : function
        Function that returns a fixture.
    someparams : anything
        Parameters of `creation_function` in correct order (no kwargs).
    """
    globals()[name] = creation_function(*someparams)
    
    
def create_standard_params(function, all_standard_params):
    """Creates parameter fixture for given function."""
    required_keys = list(inspect.signature(function).parameters)
    @pytest.fixture
    def standard_params():
        """Returns standard params needed for respective function."""
        return get_required_params(function, all_standard_params)
    
    return standard_params


def create_pos_params(function, all_pos_params):
    """Creates a fixture for all positive params of given function."""
    # all params in all_pos_params and required as function argument
    _pos_params = [param for param in all_pos_params 
                   if param in inspect.signature(function).parameters]
    
    @pytest.fixture(params=_pos_params)
    def pos_params(request):
        """Parametrizes positive parameters."""
        return request.param

    return pos_params


def get_required_params(function, all_params):
    """Checks arguments of function and returns corresponding parameters."""
    required_keys = list(inspect.signature(function).parameters)
    required_params = {k:v for k,v in all_params.items() if k in required_keys}
    return required_params


class Test_firing_rates:
    """Tests firing rates function. Probably this is a functional test."""
    
    function = staticmethod(firing_rates)
    
    # define fixtures
    standard_params_fr = inject_fixture('standard_params_fr', 
                                          create_standard_params, 
                                          firing_rates, all_standard_params)
    pos_params_fr = inject_fixture('pos_params_fr', 
                                     create_pos_params, 
                                     firing_rates, all_pos_params)
    
    def test_negative_parameters_that_should_be_positive_raise_exception(
            self, standard_params_fr, pos_params_fr):
        check_negative_parameters_that_should_be_positive_raise_exception(
            self.function, standard_params_fr, pos_params_fr)
        
    def test_V_0_larger_V_th_raise_exception(self, standard_params_fr):
        standard_params_fr['V_0_rel'] = 1.1 * standard_params_fr['V_th_rel']
        with pytest.raises(ValueError):
            self.function(**standard_params_fr)
    
    def test_correct_output(self, params_all_regimes):
        output = params_all_regimes['firing_rates']
        required_params = get_required_params(self.function, params_all_regimes)
        check_correct_output(self.function, required_params, output)
        


class Test_mean:

    # define tested function
    function = staticmethod(mean)
    
    # define fixtures
    standard_params_mean = inject_fixture('standard_params_mean', 
                                          create_standard_params, 
                                          mean, all_standard_params)
    pos_params_mean = inject_fixture('pos_params_mean', 
                                     create_pos_params, 
                                     mean, all_pos_params)
    
    def test_negative_parameters_that_should_be_positive_raise_exception(
            self, standard_params_mean, pos_params_mean):
        check_negative_parameters_that_should_be_positive_raise_exception(
            self.function, standard_params_mean, pos_params_mean)
    
    def test_correct_output(self, params_all_regimes):
        output = params_all_regimes['mean_input']
        required_params = get_required_params(self.function, params_all_regimes)
        check_correct_output(self.function, required_params, output)
        

class Test_standard_deviation:

    # define tested functiosn
    function = staticmethod(standard_deviation)
    
    # define fixtures
    standard_params_std = inject_fixture('standard_params_std', 
                                         create_standard_params, 
                                         standard_deviation, 
                                         all_standard_params)
    pos_params_std = inject_fixture('pos_params_std', 
                                    create_pos_params, 
                                    standard_deviation, 
                                    all_pos_params)
    
    def test_negative_parameters_that_should_be_positive_raise_exception(
            self, standard_params_std, pos_params_std):
        check_negative_parameters_that_should_be_positive_raise_exception(
            self.function, standard_params_std, pos_params_std)
    
    def test_correct_output(self, params_all_regimes):
        output = params_all_regimes['std_input']
        required_params = get_required_params(self.function, params_all_regimes)
        check_correct_output(self.function, required_params, output)
    
    
class Test_transfer_function:
    
    # define tested functiosn
    function = staticmethod(transfer_function)

    # define fixtures
    @pytest.fixture
    def standard_params_tf(self):
        _standard_params = get_required_params(self.function,
                                               all_standard_params)
        _standard_params['mu'] = [1*ureg.mV, 2*ureg.mV]
        _standard_params['sigma'] = [1*ureg.mV, 2*ureg.mV]
        return _standard_params
    
    def test_shift_method_is_called(self, mocker, standard_params_tf):
        mocked_tf = mocker.patch('lif_meanfield_tools.meanfield_calcs.transfer_function_1p_shift')
        standard_params_tf['method'] = 'shift'
        transfer_function(**standard_params_tf)
        mocked_tf.assert_called_once()
        
    def test_taylor_method_is_called(self, mocker, standard_params_tf):
        mocked_tf = mocker.patch('lif_meanfield_tools.meanfield_calcs.transfer_function_1p_taylor')
        standard_params_tf['method'] = 'taylor'
        transfer_function(**standard_params_tf)
        mocked_tf.assert_called_once()
            
            
class Test_transfer_function_1p_shift():
    
    # define tested functiosn
    function = staticmethod(transfer_function_1p_shift)
    
    # define fixtures
    standard_params_tf_shift = inject_fixture('standard_params_tf_shift', 
                                              create_standard_params, 
                                              transfer_function_1p_shift, 
                                              all_standard_params)
    pos_params_tf_shift = inject_fixture('pos_params_tf_shift', 
                                         create_pos_params, 
                                         transfer_function_1p_shift, 
                                         all_pos_params)
    
    def test_negative_parameters_that_should_be_positive_raise_exception(
            self, standard_params_tf_shift, pos_params_tf_shift):
        check_negative_parameters_that_should_be_positive_raise_exception(
            self.function, standard_params_tf_shift, pos_params_tf_shift)
        
    def test_warning_is_given_if_k_is_critical(self, standard_params_tf_shift):
        standard_params_tf_shift['tau_s'] = 0.5 * standard_params_tf_shift['tau_m']
        with pytest.warns(UserWarning):
            self.function(**standard_params_tf_shift)
            
    def test_exception_is_raised_if_k_is_too_large(self, standard_params_tf_shift):
        standard_params_tf_shift['tau_s'] = 2 * standard_params_tf_shift['tau_m']
        print(standard_params_tf_shift)
        with pytest.raises(ValueError):
            self.function(**standard_params_tf_shift)
    
    # def test_correct_output(self, params_all_regimes):
    #     output = params_all_regimes['tf_shift']
    #     required_params = get_required_params(self.function, params_all_regimes)
    #     check_correct_output(self.function, required_params, output)
    
        
class Test_transfer_function_1p_taylor():
    pass 
        
        

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
