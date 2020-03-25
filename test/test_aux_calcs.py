import unittest
import numpy as np

from lif_meanfield_tools.aux_calcs import *

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
    
class Test_sieger1(unittest.TestCase):
    
    
    tau_m = 10.
    tau_r = 2.
    V_th = 15
    V_0 = 0
    mu = 3
    sigma = 6
    
    def test_correct_firing_rate_prediction(self):
        result = siegert1(self.tau_m, self.tau_r, self.V_th, self.V_0   , self.mu, self.sigma)
        self.assertEqual(result, 0.0017394956677298297)
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
    
    
class TestFiringRateFunctions:
    """ Base class for testing firing rate type functoins. """
    
    # attributes are network variables like (tau_m, tau_s, V_th, V_r, ...)
    
    # common test cases 
    def test_negative_paramters_that_should_be_positive_raise_exception():
        pass
    
    def test_V_0_larger_V_th_raise_exception():
        pass
    
    def test_outside_valid_parameter_regime_raise_exception():
        pass 
    
    def test_invalid_parameter_types_raise_exception():
        pass
            
        