import unittest
from unittest import mock
from unittest.mock import patch
import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy.special import erf, zetac
from scipy.integrate import quad

import lif_meanfield_tools as lmt
from lif_meanfield_tools.aux_calcs import *

ureg = lmt.ureg

fixtures_input_path = 'tests/unit/fixtures/input/'
fixtures_output_path = 'tests/unit/fixtures/output/'


class TestFiringRateRelatedFunctions(ABC):
    """ Base class for testing firing rate type functions. """
    
    @property
    @abstractmethod
    def parameters(self):
        """ 
        Returns all standard arguments in a dictionary.
        
        NOTE: Needs to contain V_th_rel and V_0_rel! 
        Please improve this implementation if possible. I would like to enforce 
        this upon subclassing, but I don't know how. 
        """
        
        pass
    
    @property
    @abstractmethod
    def positive_params(self):
        """ List of names of positive parameters. """
        
        pass
    
    @property
    @abstractmethod
    def function(self):
        """ The function to be tested. """
        
        pass
    
    @property
    @abstractmethod
    def precision(self):
        """ 
        Precision to which rate result needs to coincide with expected result.
        """
        
        pass
    
    def integrand(self, x):
        return  np.exp(x**2) * (1 + erf(x))

    def real_siegert(self, tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma):
        """ Siegert formula as given in Fourcaud Brunel 2002 eq. 4.11 """

        y_th = (V_th_rel - mu) / sigma
        y_r = (V_0_rel - mu) / sigma
        
        nu = 1 / (tau_r + np.sqrt(np.pi) * tau_m * quad(self.integrand, y_r, y_th)[0])
        
        return nu
    
    def real_shifted_siegert(self, tau_m, tau_s, tau_r, V_th_rel, V_0_rel, mu, sigma):
        """ 
        Siegert formula with shifted boundaries for the colored noise case.
        
        Introduced in Fourcaud 2002, and Schuecker 2015.
        """
        
        alpha = np.sqrt(2.) * abs(zetac(0.5) + 1)
        k = np.sqrt(tau_s / tau_m)
        
        V_th_eff = V_th_rel + sigma * alpha * k / 2
        V_0_eff = V_0_rel + sigma * alpha * k / 2
        
        nu = self.real_siegert(tau_m, tau_r, V_th_eff, V_0_eff, mu, sigma)
        
        return nu              
    
    def test_negative_parameters_that_should_be_positive_raise_exception(self):
        for param in self.positive_params:
            
            temp_params = self.parameters
            # make parameter negative 
            temp_params[param] *= -1
            
            with self.assertRaises(ValueError):
                self.function(**temp_params)
            
    def test_V_0_larger_V_th_raise_exception(self):
        """ 
        WARNING: expect the subclass parameters to contain V_th_rel and V_0_rel.
        """
        
        temp_params = self.parameters
        temp_params['V_th_rel'] = 0 * ureg.mV
        temp_params['V_0_rel'] = 1 * ureg.mV
        
        with self.assertRaises(ValueError):
            self.function(**temp_params)


class TestFiringRateFunctions(TestFiringRateRelatedFunctions):
    
    @abstractmethod
    def expected_output(self):
        """ Function for calculating the expected result. """
        
        pass

    def check_output_for_given_params(self, params):
        """ 
        Calc expected output and result and assert equality for given parameters.
        """
        
        temp_params = self.parameters.copy()
        temp_params.update(params)                                
                        
        expected_output = self.expected_output(**temp_params)
        
        result = self.function(**temp_params)
        
        self.assertAlmostEqual(expected_output, result, self.precision)


class TestFiringRateWhiteNoiseCase(TestFiringRateFunctions):
    
    @property
    def positive_params(self):
        return ['tau_m', 'tau_r', 'sigma']
    
    def expected_output(self, **parameters):
        return self.real_siegert(**parameters)
        

class Test_siegert1(unittest.TestCase, TestFiringRateWhiteNoiseCase):
    
    @classmethod
    def setUpClass(cls):
        cls._parameters_noise_driven_regime = np.load(fixtures_input_path + 'white_noise_firing_rate_functions_noise_driven_regime.npy')
        cls._parameters_negative_firing_rate_regime = np.load(fixtures_input_path + 'white_noise_firing_rate_functions_negative_firing_rate_regime.npy')
        
    def setUp(self):
        self.tau_m = 10. * ureg.ms
        self.tau_r = 2 * ureg.ms
        self.V_th_rel = 15 * ureg.mV
        self.V_0_rel = 0 * ureg.mV
        self.mu = 3 * ureg.mV
        self.sigma = 6 * ureg.mV    
        
    @property
    def parameters(self):
        return dict(tau_m = self.tau_m,
                    tau_r = self.tau_r,
                    V_th_rel = self.V_th_rel,
                    V_0_rel = self.V_0_rel,
                    mu = self.mu,
                    sigma = self.sigma)

    @property
    def function(self):
        return siegert1

    @property
    def precision(self):
        return 10
    
    @property
    def parameters_noise_driven_regime(self):
        return self._parameters_noise_driven_regime
    
    @property
    def parameters_negative_firing_rate_regime(self):
        return self._parameters_noise_driven_regime
    
    def test_mu_larger_V_th_raises_exception(self):
        """ Give warning if mu > V_th, Siegert2 should be used. """

        temp_params = self.parameters
        temp_params['mu'] = temp_params['V_th_rel'] * 1.1

        with self.assertRaises(ValueError):
            siegert1(**temp_params)        
            
    def test_correct_output_in_noise_driven_regime(self):
        for params in self.parameters_noise_driven_regime:
            self.check_output_for_given_params(params)    
            
    def test_correct_output_in_regime_where_negative_firing_rates_once_occurred(self):
        for params in self.parameters_negative_firing_rate_regime:
            self.check_output_for_given_params(params)
            

class Test_siegert2(unittest.TestCase, TestFiringRateWhiteNoiseCase):

    @classmethod
    def setUpClass(cls):
        cls._parameters_mean_driven_regime = np.load(fixtures_input_path + 'white_noise_firing_rate_functions_mean_driven_regime.npy')
            
    def setUp(self):
        self.tau_m = 10. * ureg.ms
        self.tau_r = 2 * ureg.ms
        self.V_th_rel = 15 * ureg.mV
        self.V_0_rel = 0 * ureg.mV
        self.mu = 3 * ureg.mV
        self.sigma = 6 * ureg.mV    
        
    @property
    def parameters(self):
        return dict(tau_m = self.tau_m,
                    tau_r = self.tau_r,
                    V_th_rel = self.V_th_rel,
                    V_0_rel = self.V_0_rel,
                    mu = self.mu,
                    sigma = self.sigma)
        
    @property
    def function(self):
        return siegert2
    
    @property
    def precision(self):
        return 10
    
    @property
    def parameters_mean_driven_regime(self):
        return self._parameters_mean_driven_regime
    
    def test_mu_smaller_V_th_raises_exception(self):
        """ Give warning if mu < V_th, Siegert1 should be used. """

        temp_params = self.parameters
        temp_params['mu'] = temp_params['V_th_rel'] * 0.9

        with self.assertRaises(ValueError):
            siegert2(**temp_params)        
            
    def test_correct_output_in_mean_driven_regime(self):
        for params in self.parameters_mean_driven_regime:
            self.check_output_for_given_params(params)
                
            
class TestFiringRateColoredNoiseCase(TestFiringRateFunctions):
    
    @property
    def positive_params(self):
        return ['tau_m', 'tau_s', 'tau_r', 'sigma']
    
    def expected_output(self, **parameters):
        return self.real_shifted_siegert(**parameters)
                
    def test_correct_output_in_noise_driven_regime(self):
        
        with patch('lif_meanfield_tools.aux_calcs.nu_0', wraps=self.real_siegert) as mocked_nu_0:
            for params in self.parameters_noise_driven_regime:
                self.check_output_for_given_params(params)
                
    def test_correct_output_in_mean_driven_regime(self):
        
        with patch('lif_meanfield_tools.aux_calcs.nu_0', wraps=self.real_siegert) as mocked_nu_0:
            for params in self.parameters_mean_driven_regime:
                self.check_output_for_given_params(params)
                
    def test_correct_output_in_regime_where_negative_firing_rates_once_occurred(self):
        
        with patch('lif_meanfield_tools.aux_calcs.nu_0', wraps=self.real_siegert) as mocked_nu_0:
            for params in self.parameters_negative_firing_rate_regime:
                self.check_output_for_given_params(params)


class Test_nu0_fb433(unittest.TestCase, TestFiringRateColoredNoiseCase):

    @classmethod
    def setUpClass(cls):
        cls._parameters_noise_driven_regime = np.load(fixtures_input_path + 'colored_noise_firing_rate_functions_noise_driven_regime.npy')
        cls._parameters_mean_driven_regime = np.load(fixtures_input_path + 'colored_noise_firing_rate_functions_mean_driven_regime.npy')
        cls._parameters_negative_firing_rate_regime = np.load(fixtures_input_path + 'colored_noise_firing_rate_functions_negative_firing_rate_regime.npy')
        
    def setUp(self):
        self.tau_m = 10. * ureg.ms
        self.tau_s = 0.5 * ureg.ms
        self.tau_r = 2 * ureg.ms
        self.V_th_rel = 15 * ureg.mV
        self.V_0_rel = 0 * ureg.mV
        self.mu = 3 * ureg.mV
        self.sigma = 6 * ureg.mV
        
    @property
    def parameters(self):
        return dict(tau_m = self.tau_m,
                    tau_s = self.tau_s,
                    tau_r = self.tau_r,
                    V_th_rel = self.V_th_rel,
                    V_0_rel = self.V_0_rel,
                    mu = self.mu,
                    sigma = self.sigma)
        
    @property
    def function(self):
        return nu0_fb433

    @property
    def precision(self):
        return 5

    @property
    def parameters_noise_driven_regime(self):
        return self._parameters_noise_driven_regime

    @property
    def parameters_mean_driven_regime(self):
        return self._parameters_mean_driven_regime

    @property
    def parameters_negative_firing_rate_regime(self):
        return self._parameters_negative_firing_rate_regime


class Test_nu0_fb(unittest.TestCase, TestFiringRateColoredNoiseCase):

    @classmethod
    def setUpClass(cls):
        cls._parameters_noise_driven_regime = np.load(fixtures_input_path + 'colored_noise_firing_rate_functions_noise_driven_regime.npy')
        cls._parameters_mean_driven_regime = np.load(fixtures_input_path + 'colored_noise_firing_rate_functions_mean_driven_regime.npy')
        cls._parameters_negative_firing_rate_regime = np.load(fixtures_input_path + 'colored_noise_firing_rate_functions_negative_firing_rate_regime.npy')
        
    def setUp(self):
        self.tau_m = 10. * ureg.ms
        self.tau_s = 0.5 * ureg.ms
        self.tau_r = 2 * ureg.ms
        self.V_th_rel = 15 * ureg.mV
        self.V_0_rel = 0 * ureg.mV
        self.mu = 3 * ureg.mV
        self.sigma = 6 * ureg.mV
        
    @property
    def parameters(self):
        return dict(tau_m = self.tau_m,
                    tau_s = self.tau_s,
                    tau_r = self.tau_r,
                    V_th_rel = self.V_th_rel,
                    V_0_rel = self.V_0_rel,
                    mu = self.mu,
                    sigma = self.sigma)

    @property
    def function(self):
        return nu0_fb

    @property
    def precision(self):
        return 4

    @property
    def parameters_noise_driven_regime(self):
        return self._parameters_noise_driven_regime

    @property
    def parameters_mean_driven_regime(self):
        return self._parameters_mean_driven_regime

    @property
    def parameters_negative_firing_rate_regime(self):
        return self._parameters_negative_firing_rate_regime


class Test_nu_0(unittest.TestCase):
    
    def setUp(self):
        self.parameters = dict(tau_m = 10. * ureg.ms,
                               tau_r = 2 * ureg.ms,
                               V_th_rel = 15 * ureg.mV,
                               V_0_rel = 0 * ureg.mV,
                               mu = 3 * ureg.mV,
                               sigma = 6 * ureg.mV)
    
    def test_sieger1_is_called_if_mu_smaller_V_th_rel(self):
        self.parameters['mu'] = self.parameters['V_th_rel'] * 0.9
        with patch('lif_meanfield_tools.aux_calcs.siegert1') as mock_siegert1:
            nu_0(**self.parameters)
            mock_siegert1.assert_called()
            
    def test_sieger2_is_called_if_mu_bigger_V_th_rel(self):
        self.parameters['mu'] = self.parameters['V_th_rel'] * 1.1
        with patch('lif_meanfield_tools.aux_calcs.siegert2') as mock_siegert2:
            nu_0(**self.parameters)
            mock_siegert2.assert_called()
        

class Test_Phi(unittest.TestCase):
    
    def setUp(self):
        self.test_inputs = np.load(fixtures_input_path + 'Phi.npy')
        self.expected_outputs = np.load(fixtures_output_path + 'Phi.npy')

    def test_correct_predictions(self):
        result = Phi(self.test_inputs)
        np.testing.assert_almost_equal(result, self.expected_outputs, 5)
        
        
class Test_Phi_prime_mu(unittest.TestCase):
    
    def setUp(self):
        inputs = np.load(fixtures_input_path + 'Phi_prime_mu.npz')
        self.ss = inputs['ss']
        self.sigmas = inputs['sigmas']
        self.expected_outputs = np.load(fixtures_output_path + 'Phi_prime_mu.npy')

    def test_correct_predictions(self):
        for i, (s, sigma) in enumerate(zip(self.ss, self.sigmas)):
            result = Phi_prime_mu(s, sigma)
            self.assertEqual(result, self.expected_outputs[i])
            
    def test_negative_sigma_raises_error(self):
        sigma = -1 * ureg.mV
        s = 1
        with self.assertRaises(ValueError):
            Phi_prime_mu(s, sigma)
            
    def test_zero_sigma_raises_error(self):
        sigma = 0 * ureg.mV
        s = 1
        with self.assertRaises(ZeroDivisionError):
            Phi_prime_mu(s, sigma)
            
            
class TestFiringRateDerivativeFunctions(TestFiringRateRelatedFunctions):    
    
    def check_output_for_given_params(self, parameters_regime, expected_outputs):
        with patch('lif_meanfield_tools.aux_calcs.nu_0', wraps=self.real_siegert) as mocked_nu_0:
            for params, expected_output in zip(parameters_regime, expected_outputs):
                temp_params = self.parameters.copy()
                temp_params.update(params)                                                
                result = self.function(**temp_params)
                np.testing.assert_array_almost_equal(expected_output, result, self.precision)
                
    def test_zero_sigma_raises_error(self):
        self.sigma = 0 * ureg.mV
        with self.assertRaises(ZeroDivisionError):
            self.function(**self.parameters)
    
            
class Test_d_nu_d_mu(unittest.TestCase, TestFiringRateDerivativeFunctions):
    
    @classmethod
    def setUpClass(cls):
        cls.parameters_noise_driven_regime = np.load(fixtures_input_path + 'white_noise_firing_rate_functions_noise_driven_regime.npy')
        cls.parameters_mean_driven_regime = np.load(fixtures_input_path + 'white_noise_firing_rate_functions_mean_driven_regime.npy')
        cls.parameters_negative_firing_rate_regime = np.load(fixtures_input_path + 'white_noise_firing_rate_functions_negative_firing_rate_regime.npy')
        cls.expected_output_noise_driven_regime = np.load(fixtures_output_path + 'd_nu_d_mu_noise_driven_regime.npy')
        cls.expected_output_mean_driven_regime = np.load(fixtures_output_path + 'd_nu_d_mu_mean_driven_regime.npy')
        cls.expected_output_negative_firing_rate_regime = np.load(fixtures_output_path + 'd_nu_d_mu_negative_firing_rate_regime.npy')
        
    def setUp(self):
        self.tau_m = 10. * ureg.ms
        self.tau_r = 2 * ureg.ms
        self.V_th_rel = 15 * ureg.mV
        self.V_0_rel = 0 * ureg.mV
        self.mu = 3 * ureg.mV
        self.sigma = 6 * ureg.mV
        
    @property
    def parameters(self):
        return dict(tau_m = self.tau_m,
                    tau_r = self.tau_r,
                    V_th_rel = self.V_th_rel,
                    V_0_rel = self.V_0_rel,
                    mu = self.mu,
                    sigma = self.sigma)
        
    @property
    def function(self):
        return d_nu_d_mu

    @property
    def positive_params(self):
        return ['tau_m', 'tau_r', 'sigma']
    
    @property
    def precision(self):
        return 10
    
    def test_correct_output_in_noise_driven_regime(self):
        self.check_output_for_given_params(self.parameters_noise_driven_regime, self.expected_output_noise_driven_regime)

    def test_correct_output_in_mean_driven_regime(self):
        self.check_output_for_given_params(self.parameters_noise_driven_regime, self.expected_output_noise_driven_regime)
    
    def test_correct_output_in_negative_firing_rate_regime(self):
        self.check_output_for_given_params(self.parameters_noise_driven_regime, self.expected_output_noise_driven_regime)
                
        
class Test_d_nu_d_mu_fb433(unittest.TestCase, TestFiringRateDerivativeFunctions):
    
    @classmethod
    def setUpClass(cls):
        cls.parameters_noise_driven_regime = np.load(fixtures_input_path + 'colored_noise_firing_rate_functions_noise_driven_regime.npy')
        cls.parameters_mean_driven_regime = np.load(fixtures_input_path + 'colored_noise_firing_rate_functions_mean_driven_regime.npy')
        cls.parameters_negative_firing_rate_regime = np.load(fixtures_input_path + 'colored_noise_firing_rate_functions_negative_firing_rate_regime.npy')
        cls.d_nu_d_mu_input_noise_driven_regime = np.load(fixtures_input_path + 'd_nu_d_mu_fb433_d_nu_d_mu_noise_driven_regime.npy')
        cls.d_nu_d_mu_input_mean_driven_regime = np.load(fixtures_input_path + 'd_nu_d_mu_fb433_d_nu_d_mu_mean_driven_regime.npy')
        cls.d_nu_d_mu_input_negative_firing_rate_regime = np.load(fixtures_input_path + 'd_nu_d_mu_fb433_d_nu_d_mu_negative_firing_rate_regime.npy')
        cls.expected_output_noise_driven_regime = np.load(fixtures_output_path + 'd_nu_d_mu_fb433_noise_driven_regime.npy')
        cls.expected_output_mean_driven_regime = np.load(fixtures_output_path + 'd_nu_d_mu_fb433_mean_driven_regime.npy')
        cls.expected_output_negative_firing_rate_regime = np.load(fixtures_output_path + 'd_nu_d_mu_fb433_negative_firing_rate_regime.npy')
        
    def setUp(self):
        self.tau_m = 10. * ureg.ms
        self.tau_s = 0.5 * ureg.ms
        self.tau_r = 2 * ureg.ms
        self.V_th_rel = 15 * ureg.mV
        self.V_0_rel = 0 * ureg.mV
        self.mu = 3 * ureg.mV
        self.sigma = 6 * ureg.mV
        
    @property
    def parameters(self):
        return dict(tau_m = self.tau_m,
                    tau_s = self.tau_s,
                    tau_r = self.tau_r,
                    V_th_rel = self.V_th_rel,
                    V_0_rel = self.V_0_rel,
                    mu = self.mu,
                    sigma = self.sigma)
        
    @property
    def function(self):
        return d_nu_d_mu_fb433
    
    @property
    def positive_params(self):
        return ['tau_m', 'tau_s', 'tau_r', 'sigma']
    
    @property
    def precision(self):
        return 10
    
    def check_output_for_given_params(self, parameters_regime, d_nu_d_mu_input, expected_outputs):
        with patch('lif_meanfield_tools.aux_calcs.d_nu_d_mu') as mocked_d_nu_d_mu:
            mocked_d_nu_d_mu.side_effect = d_nu_d_mu_input
            super().check_output_for_given_params(parameters_regime, expected_outputs)

    def test_correct_output_in_noise_driven_regime(self):
        self.check_output_for_given_params(self.parameters_noise_driven_regime, self.d_nu_d_mu_input_noise_driven_regime, self.expected_output_noise_driven_regime)

    def test_correct_output_in_mean_driven_regime(self):
        self.check_output_for_given_params(self.parameters_mean_driven_regime, self.d_nu_d_mu_input_mean_driven_regime, self.expected_output_mean_driven_regime)
    
    def test_correct_output_in_negative_firing_rate_regime(self):
        self.check_output_for_given_params(self.parameters_negative_firing_rate_regime, self.d_nu_d_mu_input_negative_firing_rate_regime, self.expected_output_negative_firing_rate_regime)


class Test_d_nu_d_nu_in_fb(unittest.TestCase, TestFiringRateDerivativeFunctions):
    
    @classmethod
    def setUpClass(cls):
        cls.parameters_noise_driven_regime = np.load(fixtures_input_path + 'd_nu_d_nu_in_fb_noise_driven_regime.npy')
        cls.parameters_mean_driven_regime = np.load(fixtures_input_path + 'd_nu_d_nu_in_fb_mean_driven_regime.npy')
        cls.parameters_negative_firing_rate_regime = np.load(fixtures_input_path + 'd_nu_d_nu_in_fb_negative_firing_rate_regime.npy')
        cls.expected_output_noise_driven_regime = np.load(fixtures_output_path + 'd_nu_d_nu_in_fb_noise_driven_regime.npy')
        cls.expected_output_mean_driven_regime = np.load(fixtures_output_path + 'd_nu_d_nu_in_fb_mean_driven_regime.npy')
        cls.expected_output_negative_firing_rate_regime = np.load(fixtures_output_path + 'd_nu_d_nu_in_fb_negative_firing_rate_regime.npy')
        
    def setUp(self):
        self.tau_m = 10. * ureg.ms
        self.tau_s = 0.5 * ureg.ms
        self.tau_r = 2 * ureg.ms
        self.j =  0.1756 * ureg.mV
        self.V_th_rel = 15 * ureg.mV
        self.V_0_rel = 0 * ureg.mV
        self.mu = 3 * ureg.mV
        self.sigma = 6 * ureg.mV
        
    @property
    def parameters(self):
        return dict(tau_m = self.tau_m,
                    tau_s = self.tau_s,
                    tau_r = self.tau_r,
                    j = self.j,
                    V_th_rel = self.V_th_rel,
                    V_0_rel = self.V_0_rel,
                    mu = self.mu,
                    sigma = self.sigma)
        
    @property
    def function(self):
        return d_nu_d_nu_in_fb
    
    @property
    def positive_params(self):
        return ['tau_m', 'tau_s', 'tau_r', 'sigma']
    
    @property
    def precision(self):
        return 10
    
    def test_correct_output_in_noise_driven_regime(self):
        self.check_output_for_given_params(self.parameters_noise_driven_regime, self.expected_output_noise_driven_regime)

    def test_correct_output_in_mean_driven_regime(self):
        self.check_output_for_given_params(self.parameters_noise_driven_regime, self.expected_output_noise_driven_regime)
    
    def test_correct_output_in_negative_firing_rate_regime(self):
        self.check_output_for_given_params(self.parameters_noise_driven_regime, self.expected_output_noise_driven_regime)

            
class Test_Psi(unittest.TestCase):

    def setUp(self):
        inputs = np.load(fixtures_input_path + 'Psi.npz')
        self.zs = inputs['zs']
        self.xs = inputs['xs']
        self.pcfu = inputs['pcfu']
        self.expected_outputs = np.load(fixtures_output_path + 'Psi.npy')
        self.function = Psi
        
    def test_correct_output(self):
        with patch('lif_meanfield_tools.aux_calcs.mpmath.pcfu') as mock_pcfu:
            mock_pcfu.side_effect = self.pcfu
            for expected_output, z, x in zip(self.expected_outputs, self.zs, self.xs):
                result = self.function(z, x)
                self.assertEqual(expected_output, result)
    
            
class Test_d_Psi(unittest.TestCase):
    
    def setUp(self):
        inputs = np.load(fixtures_input_path + 'd_Psi.npz')
        self.zs = inputs['zs']
        self.xs = inputs['xs']
        self.psi = inputs['psi']
        self.expected_outputs = np.load(fixtures_output_path + 'd_Psi.npy')
        self.function = d_Psi
        
    def test_correct_output(self):
        with patch('lif_meanfield_tools.aux_calcs.Psi') as mock_psi:
            mock_psi.side_effect = self.psi
            for expected_output, z, x in zip(self.expected_outputs, self.zs, self.xs):
                result = self.function(z, x)
                self.assertEqual(expected_output, result)
        

class Test_d_2_Psi(unittest.TestCase):
    
    def setUp(self):
        inputs = np.load(fixtures_input_path + 'd_2_Psi.npz')
        self.zs = inputs['zs']
        self.xs = inputs['xs']
        self.psi = inputs['psi']
        self.expected_outputs = np.load(fixtures_output_path + 'd_2_Psi.npy')
        self.function = d_2_Psi
        
    def test_correct_output(self):
        with patch('lif_meanfield_tools.aux_calcs.Psi') as mock_psi:
            mock_psi.side_effect = self.psi
            for expected_output, z, x in zip(self.expected_outputs, self.zs, self.xs):
                result = self.function(z, x)
                self.assertEqual(expected_output, result)
        
    
class Test_Psi_x_r(unittest.TestCase):
    
    def setUp(self):
        self.z = 0
        self.x = 1
        self.y = 2
        
    def test_Psi_is_called_two_times(self):
        with patch('lif_meanfield_tools.aux_calcs.Psi') as mock_Psi:
            Psi_x_r(self.z, self.x, self.y)
            mock_Psi.assert_has_calls([mock.call(0,1), mock.call(0,2)], any_order=True)

            
class Test_dPsi_x_r(unittest.TestCase):
    
    def setUp(self):
        self.z = 0
        self.x = 1
        self.y = 2
        
    def test_d_Psi_is_called_two_times(self):
        with patch('lif_meanfield_tools.aux_calcs.d_Psi') as mock_Psi:
            dPsi_x_r(self.z, self.x, self.y)
            mock_Psi.assert_has_calls([mock.call(0,1), mock.call(0,2)], any_order=True)
            
            
class Test_d2Psi_x_r(unittest.TestCase):
    
    def setUp(self):
        self.z = 0
        self.x = 1
        self.y = 2
        
    def test_d_2_Psi_is_called_two_times(self):
        with patch('lif_meanfield_tools.aux_calcs.d_2_Psi') as mock_Psi:
            d2Psi_x_r(self.z, self.x, self.y)
            mock_Psi.assert_has_calls([mock.call(0,1), mock.call(0,2)], any_order=True)

        
# class Test_determinant(unittest.TestCase):
# 
#     def test_real_matrix_with_zero_determinant(self):
#         a = [1,2,3]
#         M = np.array([a,a,a])
#         result = determinant(M)
#         real_determinant = 0
#         self.assertEqual(result, real_determinant)
# 
#     def test_real_matrix_with_positive_determinant(self):
#         M = np.array([[1,2,3],[2,1,3],[3,1,2]])
#         result = determinant(M)
#         real_determinant = 6
#         self.assertEqual(result, real_determinant)
# 
#     def test_real_matrix_with_negative_determinant(self):
#         M = np.array([[1,2,3],[3,1,2],[2,1,3]])
#         result = determinant(M)
#         real_determinant = -6
#         self.assertEqual(result, real_determinant)
# 
#     def test_non_square_matrix(self):
#         M = np.array([[1,2,3],[2,3,1]])
#         with self.assertRaises(np.linalg.LinAlgError):
#             result = determinant(M)
# 
#     def test_matrix_with_imaginary_determinant(self):
#         M = np.array([[complex(0,1), 1], [0, 1]])
#         real_determinant = np.linalg.det(M)
#         result = determinant(M)
#         self.assertEqual(result, real_determinant)


class Test_determinant_same_rows(unittest.TestCase):
    """ Implement, when you know what determinant in aux_calcs is doing. """
    pass


class Test_p_hat_boxcar(unittest.TestCase):
    
    def setUp(self):
        inputs = np.load(fixtures_input_path + 'p_hat_boxcar.npz')
        self.ks = inputs['ks']
        self.widths = inputs['widths']
        self.expected_outputs = np.load(fixtures_output_path + 'p_hat_boxcar.npy')
    
    def test_zero_frequency_input(self):
        k = 0 
        width = 1
        self.assertEqual(p_hat_boxcar(k, width), 1)

    def test_zero_width_raises_exception(self):
        k = 1
        width = 0 
        with self.assertRaises(ValueError):
            p_hat_boxcar(k, width)

    def test_negative_width_raises_exception(self):
        k = 1
        width = -1
        with self.assertRaises(ValueError):
            p_hat_boxcar(k, width)
    
    def test_correct_output(self):
        for expected_output, k, width in zip(self.expected_outputs, self.ks, self.widths):
            result = p_hat_boxcar(k, width)
            self.assertEqual(expected_output, result)
            
            
class Test_solve_chareq_rate_boxcar(unittest.TestCase):
    """ Implement, when you know what determinant in aux_calcs is doing. """
    pass