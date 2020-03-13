import unittest
import numpy as np

from lif_meanfield_tools.aux_calcs import *

class test_determinant(unittest.TestCase):
    
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
        