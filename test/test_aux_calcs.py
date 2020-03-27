import unittest
import numpy as np
from scipy.special import erf 
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


class Test_nu0_fb433(unittest.TestCase):
    
    # nu0_fb433(tau_m, tau_s, tau_r, V_th_rel, V_0_rel, mu, sigma)    
    
    def test_correct_firing_rate_prediction(self):
        # test several parameter combinations for correct prediction
        pass 
    
    def test_negative_paramters_that_should_be_positive_raise_exception(self):
        pass
    
    def test_V_0_larger_V_th_raise_exception(self):
        pass
    
    def test_outside_valid_parameter_regime_raise_exception(self):
        pass 
    
    def test_invalid_parameter_types_raise_exception(self):
        pass
    
    
class Test_siegert1(unittest.TestCase):
    
    
    def setUp(self):
        self.tau_m = 10. * ureg.ms
        self.tau_r = 2. * ureg.ms
        self.V_th = 15 * ureg.mV
        self.V_0 = 0 * ureg.mV
        self.mu = 3 * ureg.mV
        self.sigma = 6 * ureg.mV


    def integrand(self, x):
        return  np.exp(x**2) * (1 + erf(x))
        
        
    def real_siegert(self, mu, sigma, V_th, V_r, tau_m, tau_r):
        """ Siegert formula as given in Fourcaud Brunel 2002 eq. 4.11 """
        
        y_th = (V_th - mu) / sigma
        y_r = (V_r - mu) / sigma
        
        nu = 1 / (tau_r + np.sqrt(np.pi) * tau_m * quad(self.integrand, y_r, y_th)[0])
        
        return nu
    
    
    def test_correct_firing_rate_prediction(self):
        
        # values taken from microcircuit minimal example
        mus = [3.30031035, 7.02709379, 7.18151477, 9.18259078] * ureg.mV
        sigmas = [6.1901737, 5.11420662, 5.96478947, 4.89397196] * ureg.mV

        for mu, sigma in zip(mus, sigmas):
            expected_result = self.real_siegert(mu, sigma, self.V_th, self.V_0, self.tau_m, self.tau_r)
            result = siegert1(self.tau_m, self.tau_r, self.V_th, self.V_0, mu, sigma)
            
            self.assertAlmostEqual(expected_result, result, 10)
            
        # these values occured when we got negative firing rates from nu0_fb433
        self.tau_m = 20 * ureg.ms
        mus = [-4.69428276, -12.88765852, -21.41462729, 6.76113423] * ureg.mV
        sigmas = [13.51676476, 9.26667293, 10.42112985, 4.56041] * ureg.mV

        for mu, sigma in zip(mus, sigmas):
            expected_result = self.real_siegert(mu, sigma, self.V_th, self.V_0, self.tau_m, self.tau_r)
            result = siegert1(self.tau_m, self.tau_r, self.V_th, self.V_0, mu, sigma)
            
            self.assertAlmostEqual(expected_result, result, 10)
        
    
    def test_negative_paramters_that_should_be_positive_raise_exception(self):
        
        params = ['tau_m', 'tau_r', 'sigma']
        
        for i, param in enumerate(params):
            
            # make parameter negative 
            setattr(self, param, getattr(self, param)*(-1))
            
            with self.assertRaises(ValueError):
                siegert1(self.tau_m, self.tau_r, self.V_th, self.V_0, self.mu, self.sigma)
            
            # reset parameter
            setattr(self, param, getattr(self, param)*(-1))
            
            
    def test_V_0_larger_V_th_raises_exception(self):
    
        self.V_th_temp = 0 * ureg.mV
        self.V_0_temp = 1 * ureg.mV
        
        with self.assertRaises(ValueError):
            siegert1(self.tau_m, self.tau_r, self.V_th_temp, self.V_0_temp, self.mu, self.sigma)
        
    
    def test_mu_larger_V_th_raises_exception(self):
        """ Give warning if mu > V_th, Siegert2 should be used. """
        
        self.mu = self.V_th * 1.1
        
        with self.assertRaises(ValueError):
            siegert1(self.tau_m, self.tau_r, self.V_th, self.V_0, self.mu, self.sigma)
        
        
    
# 
# class TestFiringRateFunctions:
#     """ Base class for testing firing rate type functoins. """
# 
#     # attributes are network variables like (tau_m, tau_s, V_th, V_r, ...)
# 
#     # common test cases 
#     def test_negative_paramters_that_should_be_positive_raise_exception():
#         pass
# 
#     def test_V_0_larger_V_th_raise_exception():
#         pass
# 
#     def test_outside_valid_parameter_regime_raise_exception():
#         pass 
# 
#     def test_invalid_parameter_types_raise_exception():
#         pass
# 
# 