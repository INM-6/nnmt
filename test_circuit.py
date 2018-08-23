"""test_circuit.py: tests of major functions of circuit.
Test data created using numpy 1.8.2.

Authors: Hannah Bos, Jannis Schuecker
"""

import circuit
import numpy as np
import h5py_wrapper.wrapper as h5
import unittest

class TestCircuit(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.rtol = 1e-12
        self.atol = 1e-7
        self.test_data = 'test_data.h5'
        # calculate static and dynamic transfer function only once
        self.net_init = circuit.Circuit('microcircuit', from_file=False)

    def setUp(self):
        self.net_stat = circuit.Circuit('microcircuit', 
                                        analysis_type='stationary')
        self.net = circuit.Circuit('microcircuit')
   
    def testAlterParams(self):
        net = circuit.Circuit('microcircuit', analysis_type='stationary')
        # alteration of the membrane potential
        self.net_stat.alter_params({'taum': 11.0})
        self.assertEqual(self.net_stat.params['taum'], 11.0)
        self.assertEqual(self.net_stat.ana.taum, 11.0)
        # test altering the weight matrix W
        W_new = self.net_stat.params['W']
        W_new[2][3] *= 0.9
        self.net_stat.alter_params({'W': W_new})
        assert(np.allclose(W_new,self.net_stat.params['W'], 
                           rtol=self.rtol, atol=self.atol))
        assert(np.allclose(W_new, self.net_stat.ana.W, 
                           rtol=self.rtol, atol=self.atol))

    def testFiringRates(self):
        firing_rates = h5.load(self.test_data, 'firing_rates')
        assert(np.allclose(firing_rates, self.net_stat.th_rates, 
                           rtol=self.rtol, atol=self.atol))

    def testTransferFunction(self):
        freqs, dtf = self.net.get_transfer_function()
        trans_func_test = h5.load(self.test_data, '/dyn_trans_func')
        assert(np.allclose(dtf, trans_func_test, rtol=self.rtol, atol=self.atol))

    def testCreatePowerSpectra(self):
        for delay_dist in ['none','gaussian','truncated_gaussian']:
            self.net.alter_params({'delay_dist': delay_dist})
            freqs, power = self.net.create_power_spectra()
            power_test = h5.load(self.test_data, delay_dist + '/power')
            assert (np.allclose(power_test, power, rtol=self.rtol, atol=self.atol))

    def testCreateEigenvalueSpectra(self):
        for delay_dist in ['none','gaussian','truncated_gaussian']:
            self.net.alter_params({'delay_dist': delay_dist})
            freqs, eigs = self.net.create_eigenvalue_spectra('MH')
            eigs_test = h5.load(self.test_data, delay_dist + '/eigs')
            assert (np.allclose(eigs_test, eigs, rtol=self.rtol, atol=self.atol))

    def testGetSensitivityMeasure(self):
        freq = 82.0
        indices = [None] + [i for i in range(8)]
        for delay_dist in ['none','gaussian','truncated_gaussian']:
            self.net.alter_params({'delay_dist': delay_dist})
            for i in indices:
                T = self.net.get_sensitivity_measure(freq, index=i)
                T_test = h5.load(self.test_data, delay_dist + '/' + str(i) + '/T')
                assert (np.allclose(T_test, T, rtol=self.rtol, atol=self.atol))

    def testEmpiricalTransferFunction(self):
        params = {}
        params['tf_mode'] = 'empirical' 
        params['tau_impulse'] = np.array([8.555, 5.611, 4.167, 4.381, 4.131, 3.715, 4.538, 3.003])
        params['delta_f'] = np.array([0.0880, 0.458, 0.749, 0.884, 1.183, 1.671, 0.140, 1.710])/self.net.params['w']
        net = circuit.Circuit('microcircuit', params)
        freqs, power = net.create_power_spectra()
        power_test = h5.load(self.test_data, 'empirical/power')
        assert (np.allclose(power_test, power, rtol=self.rtol, atol=self.atol))

if __name__ == '__main__':
    unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCircuit)
    unittest.TextTestRunner(verbosity=2).run(suite)
