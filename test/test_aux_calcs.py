import unittest
import warnings
from abc import ABC, abstractmethod
import numpy as np
from scipy.special import erf, zetac
from scipy.integrate import quad

import lif_meanfield_tools as lmt
from lif_meanfield_tools.aux_calcs import *

ureg = lmt.ureg

class Test_determinant(unittest.TestCase):

    def test_real_matrix_with_zero_determinant(self):
        a = [1,2,3]
        M = np.array([a,a,a])
        result = determinant(M)
        real_determinant = 0
        self.assertEqual(result, real_determinant)

    def test_real_matrix_with_positive_determinant(self):
        M = np.array([[1,2,3],[2,1,3],[3,1,2]])
        result = determinant(M)
        real_determinant = 6
        self.assertEqual(result, real_determinant)

    def test_real_matrix_with_negative_determinant(self):
        M = np.array([[1,2,3],[3,1,2],[2,1,3]])
        result = determinant(M)
        real_determinant = -6
        self.assertEqual(result, real_determinant)

    def test_non_square_matrix(self):
        M = np.array([[1,2,3],[2,3,1]])
        with self.assertRaises(np.linalg.LinAlgError):
            result = determinant(M)

    def test_matrix_with_imaginary_determinant(self):
        M = np.array([[complex(0,1), 1], [0, 1]])
        real_determinant = np.linalg.det(M)
        result = determinant(M)
        self.assertEqual(result, real_determinant)



class TestFiringRateFunctions(ABC):
    """ Base class for testing firing rate type functions. """
    
    @property
    @abstractmethod
    def parameters_for_firing_rate_test(self):
        # expect list of parameter dictionaries
        pass
    
    @property
    @abstractmethod
    def positive_params(self):
        pass
    
    @property
    @abstractmethod
    def function(self):
        pass
    
    @property
    @abstractmethod
    def precision(self):
        pass
    
    
    @abstractmethod
    def expected_firing_rate(self):
        pass
    
    
    def integrand(self, x):
        return  np.exp(x**2) * (1 + erf(x))


    def real_siegert(self, mu, sigma, V_th_rel, V_0_rel, tau_m, tau_r):
        """ Siegert formula as given in Fourcaud Brunel 2002 eq. 4.11 """

        y_th = (V_th_rel - mu) / sigma
        y_r = (V_0_rel - mu) / sigma

        nu = 1 / (tau_r + np.sqrt(np.pi) * tau_m * quad(self.integrand, y_r, y_th)[0])

        return nu
    
    
    def test_negative_paramters_that_should_be_positive_raise_exception(self):
        
        
        for param in self.positive_params:
            
            # make parameter negative 
            self.parameters[param] *= -1
            
            with self.assertRaises(ValueError):
                self.function(**self.parameters)
            
            # make parameter positive again
            self.parameters[param] *= -1
            

    def test_V_0_larger_V_th_raise_exception(self):
        
        self.parameters['V_th_rel'] = 0 * ureg.mV
        self.parameters['V_0_rel'] = 1 * ureg.mV
        
        
        with self.assertRaises(ValueError):
            self.function(**self.parameters)
        
    
    def test_correct_firing_rate_prediction(self):

        for params in self.parameters_for_firing_rate_test:
        
            temp_params = self.parameters.copy()
            temp_params.update(params)                                                
            expected_result = self.expected_firing_rate(**temp_params)
            result = self.function(**temp_params)
            self.assertAlmostEqual(expected_result, result, self.precision)



class Test_siegert1(unittest.TestCase, TestFiringRateFunctions):
    
    @classmethod
    def setUpClass(cls):
                
        V_th = 17.85865096129104 * ureg.mV
        mus = [3.30031035, 7.02709379, 7.18151477, 9.18259078] * ureg.mV
        sigmas = [6.1901737, 5.11420662, 5.96478947, 4.89397196] * ureg.mV
        cls.parameters_for_firing_rate_test_0 = [dict(mu=mu, sigma=sigma, V_th_rel=V_th) for mu, sigma in zip(mus, sigmas)]
        
        tau_m = 20 * ureg.ms
        mus = [-4.69428276, -12.88765852, -21.41462729, 6.76113423] * ureg.mV
        sigmas = [13.51676476, 9.26667293, 10.42112985, 4.56041] * ureg.mV
        cls.parameters_for_firing_rate_test_1 = [dict(mu=mu, sigma=sigma, tau_m=tau_m) for mu, sigma in zip(mus, sigmas)]
        
        
    def setUp(self):
        self.parameters = dict(tau_m = 10. * ureg.ms,
                               tau_r = 2 * ureg.ms,
                               V_th_rel = 15 * ureg.mV,
                               V_0_rel = 0 * ureg.mV,
                               mu = 3 * ureg.mV,
                               sigma = 6 * ureg.mV)
        
        
    @property
    def function(self):
        return siegert1
    
    
    @property
    def precision(self):
        return 10
    
    
    @property
    def positive_params(self):
        return ['tau_m', 'tau_r', 'sigma']
    
    
    @property
    def parameters_for_firing_rate_test(self):
        return self.parameters_for_firing_rate_test_0 + self.parameters_for_firing_rate_test_1


    def expected_firing_rate(self, **parameters):
        return self.real_siegert(**parameters)

    
    def test_mu_larger_V_th_raises_exception(self):
        """ Give warning if mu > V_th, Siegert2 should be used. """

        self.parameters['mu'] = self.parameters['V_th_rel'] * 1.1

        with self.assertRaises(ValueError):
            siegert1(**self.parameters)


class Test_siegert2(unittest.TestCase, TestFiringRateFunctions):

    @classmethod
    def setUpClass(cls):
                
        tau_m = 20. * ureg.ms
        tau_r = 0.5 * ureg.ms
        V_th = 20 * ureg.mV
        mus =  [741.89455754, 21.24112709, 35.86521795, 40.69297877, 651.19761921] * ureg.mV
        sigmas = [39.3139564, 6.17632725, 9.79196704, 10.64437979, 37.84928217] * ureg.mV
        cls.parameters_for_firing_rate_test_0 = [dict(mu=mu, sigma=sigma, V_th_rel=V_th, tau_m=tau_m, tau_r=tau_r) for mu, sigma in zip(mus, sigmas)]
        
        
    def setUp(self):
        self.parameters = dict(tau_m = 10. * ureg.ms,
                               tau_r = 2 * ureg.ms,
                               V_th_rel = 15 * ureg.mV,
                               V_0_rel = 0 * ureg.mV,
                               mu = 3 * ureg.mV,
                               sigma = 6 * ureg.mV)
        
    @property
    def function(self):
        return siegert2


    @property
    def precision(self):
        return 10


    @property
    def positive_params(self):
        return ['tau_m', 'tau_r', 'sigma']


    @property
    def parameters_for_firing_rate_test(self):
        return self.parameters_for_firing_rate_test_0
    
    
    def expected_firing_rate(self, **parameters):
        return self.real_siegert(**parameters)


    def test_mu_smaller_V_th_raises_exception(self):
        """ Give warning if mu < V_th, Siegert1 should be used. """

        self.parameters['mu'] = self.parameters['V_th_rel'] * 0.9

        with self.assertRaises(ValueError):
            siegert2(**self.parameters)



class Test_nu0_fb433(unittest.TestCase, TestFiringRateFunctions):

    @classmethod
    def setUpClass(cls):
        
        tau_s = 2. * ureg.ms
        mus = [3.30031035, 7.02709379, 7.18151477, 9.18259078] * ureg.mV
        sigmas = [6.1901737, 5.11420662, 5.96478947, 4.89397196] * ureg.mV
        cls.parameters_for_firing_rate_test_0 = [dict(mu=mu, sigma=sigma, tau_s=tau_s) for mu, sigma in zip(mus, sigmas)]
        
        tau_m = 20 * ureg.ms
        mus = [-4.69428276, -12.88765852, -21.41462729, 6.76113423] * ureg.mV
        sigmas = [13.51676476, 9.26667293, 10.42112985, 4.56041] * ureg.mV
        cls.parameters_for_firing_rate_test_1 = [dict(mu=mu, sigma=sigma, tau_m=tau_m) for mu, sigma in zip(mus, sigmas)]

        tau_m = 20. * ureg.ms
        tau_r = 0.5 * ureg.ms
        V_th = 20 * ureg.mV
        mus =  [741.89455754, 651.19761921, 21.24112709, 35.86521795, 40.69297877] * ureg.mV
        sigmas = [39.3139564, 37.84928217, 6.17632725, 9.79196704, 10.64437979] * ureg.mV
        cls.parameters_for_firing_rate_test_2 = [dict(mu=mu, sigma=sigma, tau_m=tau_m, tau_r=tau_r, V_th=V_th) for mu, sigma in zip(mus, sigmas)]
    
        
    def setUp(self):
        tau_s = 0.5 * ureg.ms,
        self.parameters = dict(tau_m = 10. * ureg.ms,
                               tau_s = 0.5 * ureg.ms,
                               tau_r = 2 * ureg.ms,
                               V_th_rel = 15 * ureg.mV,
                               V_0_rel = 0 * ureg.mV,
                               mu = 3 * ureg.mV,
                               sigma = 6 * ureg.mV)
        
        
    @property
    def function(self):
        return nu0_fb433


    @property
    def precision(self):
        return 4


    @property
    def positive_params(self):
        return ['tau_m', 'tau_s', 'tau_r', 'sigma']


    @property
    def parameters_for_firing_rate_test(self):
        return self.parameters_for_firing_rate_test_0 + self.parameters_for_firing_rate_test_1 + self.parameters_for_firing_rate_test_2
    
    
    def expected_firing_rate(self, mu, sigma, V_th_rel, V_0_rel, tau_m, tau_s, tau_r):

        alpha = np.sqrt(2.) * abs(zetac(0.5) + 1)
        k = np.sqrt(tau_s / tau_m)

        V_th_eff = V_th_rel + sigma * alpha * k / 2
        V_0_eff = V_0_rel + sigma * alpha * k / 2

        nu = self.real_siegert(mu, sigma, V_th_eff, V_0_eff, tau_m, tau_r)

        return nu              


    def test_warning_is_given_if_k_might_be_too_large(self):

        self.parameters['tau_m'] = 1 * ureg.ms
        self.parameters['tau_s'] = 0.9 * ureg.ms

        with self.assertWarns(Warning) as w:
            self.function(**self.parameters)


    def test_error_is_raised_if_k_is_too_large(self):

        self.parameters['tau_m'] = 1 * ureg.ms
        self.parameters['tau_s'] = 1.1 * ureg.ms

        with self.assertRaises(ValueError):
            self.function(**self.parameters)
