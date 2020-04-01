import unittest
from unittest.mock import patch
import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy.special import erf, zetac
from scipy.integrate import quad

import lif_meanfield_tools as lmt
from lif_meanfield_tools.aux_calcs import *

ureg = lmt.ureg
# 
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



class TestFiringRateFunctions(ABC):
    """ Base class for testing firing rate type functions. """
    
    @property
    @abstractmethod
    def parameters(self):
        """ 
        Returns all arguments in a dictionary.
        
        NOTE: Needs to contain V_th_rel and V_0_rel! 
        Please improve this implementation if possible. I would like to enforce 
        this upon subclassing, but I don't know how. 
        """
        pass
    
    @property
    @abstractmethod
    def parameters_for_firing_rate_test(self):
        """ 
        List of dictionaries containing parameters to be updated for each test.
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
    
    
    @abstractmethod
    def expected_firing_rate(self):
        """ Function for calculating the expected result. """
        pass
    
    
    def integrand(self, x):
        return  np.exp(x**2) * (1 + erf(x))


    def real_siegert(self, mu, sigma, V_th_rel, V_0_rel, tau_m, tau_r):
        """ Siegert formula as given in Fourcaud Brunel 2002 eq. 4.11 """

        y_th = (V_th_rel - mu) / sigma
        y_r = (V_0_rel - mu) / sigma

        nu = 1 / (tau_r + np.sqrt(np.pi) * tau_m * quad(self.integrand, y_r, y_th)[0])

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
        
    
    def test_correct_firing_rate_prediction(self):

        for params in self.parameters_for_firing_rate_test:
        
            temp_params = self.parameters.copy()
            temp_params.update(params)                                                
            expected_result = self.expected_firing_rate(**temp_params)
            result = self.function(**temp_params)
            self.assertAlmostEqual(expected_result, result, self.precision)



class TestFiringRateWhiteNoiseCase(TestFiringRateFunctions):
    
    @property
    def positive_params(self):
        return ['tau_m', 'tau_r', 'sigma']
    
    
    def expected_firing_rate(self, **parameters):
        return self.real_siegert(**parameters)
    


class Test_siegert1(unittest.TestCase, TestFiringRateWhiteNoiseCase):
    
    @classmethod
    def setUpClass(cls):
                
        V_th = 17.85865096129104 * ureg.mV
        mus = [3.30031035, 7.02709379, 7.18151477, 9.18259078] * ureg.mV
        sigmas = [6.1901737, 5.11420662, 5.96478947, 4.89397196] * ureg.mV
        cls._parameters_for_firing_rate_test_0 = [dict(mu=mu, sigma=sigma, V_th_rel=V_th) for mu, sigma in zip(mus, sigmas)]
        
        tau_m = 20 * ureg.ms
        mus = [-4.69428276, -12.88765852, -21.41462729, 6.76113423] * ureg.mV
        sigmas = [13.51676476, 9.26667293, 10.42112985, 4.56041] * ureg.mV
        cls._parameters_for_firing_rate_test_1 = [dict(mu=mu, sigma=sigma, tau_m=tau_m) for mu, sigma in zip(mus, sigmas)]
        
        
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
    def parameters_for_firing_rate_test(self):
        return self._parameters_for_firing_rate_test_0 + self._parameters_for_firing_rate_test_1
    
    
    def test_mu_larger_V_th_raises_exception(self):
        """ Give warning if mu > V_th, Siegert2 should be used. """

        temp_params = self.parameters
        temp_params['mu'] = temp_params['V_th_rel'] * 1.1

        with self.assertRaises(ValueError):
            siegert1(**temp_params)
            


class Test_siegert2(unittest.TestCase, TestFiringRateWhiteNoiseCase):

    @classmethod
    def setUpClass(cls):
                
        tau_m = 20. * ureg.ms
        tau_r = 0.5 * ureg.ms
        V_th = 20 * ureg.mV
        mus =  [741.89455754, 21.24112709, 35.86521795, 40.69297877, 651.19761921] * ureg.mV
        sigmas = [39.3139564, 6.17632725, 9.79196704, 10.64437979, 37.84928217] * ureg.mV
        cls._parameters_for_firing_rate_test_0 = [dict(mu=mu, sigma=sigma, V_th_rel=V_th, tau_m=tau_m, tau_r=tau_r) for mu, sigma in zip(mus, sigmas)]
        
        
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
    def parameters_for_firing_rate_test(self):
        return self._parameters_for_firing_rate_test_0
    

    def test_mu_smaller_V_th_raises_exception(self):
        """ Give warning if mu < V_th, Siegert1 should be used. """

        temp_params = self.parameters
        temp_params['mu'] = temp_params['V_th_rel'] * 0.9

        with self.assertRaises(ValueError):
            siegert2(**temp_params)
            
            
            
class TestFiringRateColoredNoiseCase(TestFiringRateFunctions):
    
    @property
    def positive_params(self):
        return ['tau_m', 'tau_s', 'tau_r', 'sigma']


    def expected_firing_rate(self, mu, sigma, V_th_rel, V_0_rel, tau_m, tau_s, tau_r):

        alpha = np.sqrt(2.) * abs(zetac(0.5) + 1)
        k = np.sqrt(tau_s / tau_m)

        V_th_eff = V_th_rel + sigma * alpha * k / 2
        V_0_eff = V_0_rel + sigma * alpha * k / 2

        nu = self.real_siegert(mu, sigma, V_th_eff, V_0_eff, tau_m, tau_r)

        return nu              


class Test_nu0_fb433(unittest.TestCase, TestFiringRateColoredNoiseCase):

    @classmethod
    def setUpClass(cls):
        
        tau_s = 2. * ureg.ms
        mus = [3.30031035, 7.02709379, 7.18151477, 9.18259078] * ureg.mV
        sigmas = [6.1901737, 5.11420662, 5.96478947, 4.89397196] * ureg.mV
        cls._parameters_for_firing_rate_test_0 = [dict(mu=mu, sigma=sigma, tau_s=tau_s) for mu, sigma in zip(mus, sigmas)]
        
        tau_m = 20 * ureg.ms
        mus = [-4.69428276, -12.88765852, -21.41462729, 6.76113423] * ureg.mV
        sigmas = [13.51676476, 9.26667293, 10.42112985, 4.56041] * ureg.mV
        cls._parameters_for_firing_rate_test_1 = [dict(mu=mu, sigma=sigma, tau_m=tau_m) for mu, sigma in zip(mus, sigmas)]

        tau_m = 20. * ureg.ms
        tau_r = 0.5 * ureg.ms
        V_th_rel = 20 * ureg.mV
        mus =  [741.89455754, 651.19761921, 21.24112709, 35.86521795, 40.69297877] * ureg.mV
        sigmas = [39.3139564, 37.84928217, 6.17632725, 9.79196704, 10.64437979] * ureg.mV
        cls._parameters_for_firing_rate_test_2 = [dict(mu=mu, sigma=sigma, tau_m=tau_m, tau_r=tau_r, V_th_rel=V_th_rel) for mu, sigma in zip(mus, sigmas)]
    
        
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
        return 4


    @property
    def parameters_for_firing_rate_test(self):
        return self._parameters_for_firing_rate_test_0 + self._parameters_for_firing_rate_test_1 + self._parameters_for_firing_rate_test_2



class Test_nu0_fb(unittest.TestCase, TestFiringRateColoredNoiseCase):

    @classmethod
    def setUpClass(cls):
        
        tau_s = 2. * ureg.ms
        mus = [3.30031035, 7.02709379, 7.18151477, 9.18259078] * ureg.mV
        sigmas = [6.1901737, 5.11420662, 5.96478947, 4.89397196] * ureg.mV
        cls._parameters_for_firing_rate_test_0 = [dict(mu=mu, sigma=sigma, tau_s=tau_s) for mu, sigma in zip(mus, sigmas)]
        
        tau_m = 20 * ureg.ms
        mus = [-4.69428276, -12.88765852, -21.41462729, 6.76113423] * ureg.mV
        sigmas = [13.51676476, 9.26667293, 10.42112985, 4.56041] * ureg.mV
        cls._parameters_for_firing_rate_test_1 = [dict(mu=mu, sigma=sigma, tau_m=tau_m) for mu, sigma in zip(mus, sigmas)]

        tau_m = 20. * ureg.ms
        tau_r = 0.5 * ureg.ms
        V_th = 20 * ureg.mV
        mus =  [741.89455754, 651.19761921, 21.24112709, 35.86521795, 40.69297877] * ureg.mV
        sigmas = [39.3139564, 37.84928217, 6.17632725, 9.79196704, 10.64437979] * ureg.mV
        cls._parameters_for_firing_rate_test_2 = [dict(mu=mu, sigma=sigma, tau_m=tau_m, tau_r=tau_r, V_th_rel=V_th) for mu, sigma in zip(mus, sigmas)]
    
        
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
    def parameters_for_firing_rate_test(self):
        return self._parameters_for_firing_rate_test_0 + self._parameters_for_firing_rate_test_1 + self._parameters_for_firing_rate_test_2



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
        
