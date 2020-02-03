# -*- coding: utf-8 -*-
"""
Integration tests reproducing data for the Figure 4 of the following
publication:

Schuecker, J., Diesmann, M. & Helias, M.
Modulated escape from a metastable state driven by colored noise.
Phys. Rev. E - Stat. Nonlinear, Soft Matter Phys. 92, 1–11 (2015).
"""

#TODO: remove path hardcoding!
import sys
sys.path.append('/home/essink/working_environment/lif_meanfield_tools')
import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from collections import defaultdict

import numpy as np

import lif_meanfield_tools as lmt
from lif_meanfield_tools.__init__ import ureg
import h5py_wrapper.wrapper as h5


class SchueckerTestCase(unittest.TestCase):
    def setUp(self):

        # Load ground truth data
        path_to_data = '/home/essink/working_environment/lif_meanfield_tools/lif_meanfield_tools/tests/integration/fixtures/'
        self.ground_truth_result = h5.load(path_to_data + 'Schuecker2015_data.h5')

        # test=10*ureg.meter
        # print('test',id(test._REGISTRY))
        #
        # @ureg.wraps(ureg.Hz / ureg.mV, (ureg.mV, ureg.mV, ureg.s, ureg.s,
        #                                 ureg.s, ureg.mV, ureg.mV, ureg.Hz))
        # def transfer_function_dummy(mu, sigma, tau_m, tau_s, tau_r, V_th_rel,
        #                             V_0_rel, omega):
        #     return omega
        #
        # print('dummy',
        #     transfer_function_dummy(1 * ureg.mV, 1 * ureg.mV, 1 * ureg.s,
        #                             1 * ureg.s, 1 * ureg.s, 1 * ureg.mV,
        #                             1 * ureg.mV, 1 * ureg.Hz))
        # print('_',
        #     lmt.meanfield_calcs._transfer_function_1p_shift(1,1,1,1,1,1,1,1))

        # Generate test data
        self.network_params = lmt.input_output.load_params(
            'lif_meanfield_tools/tests/integration/fixtures/Schuecker2015_parameters.yaml'
        )

        self.frequencies = np.logspace(
            self.network_params['f_start_exponent']['val'],
            self.network_params['f_end_exponent']['val'],
            self.network_params['n_freqs']['val']) * ureg.Hz

        self.omegas = 2 * np.pi * self.frequencies
        self.network_params['dimension'] = 1

        self.test_results = defaultdict(str)
        self.test_results['sigma'] = defaultdict(dict)

        sigma = self.network_params['sigma_1']
        self.test_results['sigma'][sigma.magnitude]['mu'] = defaultdict(dict)

        for mu in self.network_params['mean_input_1']:
        # TODO: fix pint wraps issues!
            transfer_function = [lmt.meanfield_calcs._transfer_function_1p_shift(
                mu.magnitude,
                sigma.magnitude,
                self.network_params['tau_m'].to(ureg.s).magnitude,
                self.network_params['tau_s'].to(ureg.s).magnitude,
                self.network_params['tau_r'].to(ureg.s).magnitude,
                self.network_params['theta'].magnitude,
                self.network_params['V_reset'].magnitude,
                omega.magnitude) for omega in self.omegas]

            self.test_results['sigma'][sigma.magnitude]['mu'][mu.magnitude] = {
                'absolute_value': np.abs(transfer_function),
                'phase': np.angle(transfer_function)}

        #TODO: loop more nicely over the two sigmas and the mu's
        sigma = self.network_params['sigma_2']
        self.test_results['sigma'][sigma.magnitude]['mu'] = defaultdict(dict)

        for mu in self.network_params['mean_input_2']:
        # TODO: fix pint wraps issues!
            transfer_function = [lmt.meanfield_calcs._transfer_function_1p_shift(
                mu.magnitude,
                sigma.magnitude,
                self.network_params['tau_m'].to(ureg.s).magnitude,
                self.network_params['tau_s'].to(ureg.s).magnitude,
                self.network_params['tau_r'].to(ureg.s).magnitude,
                self.network_params['theta'].magnitude,
                self.network_params['V_reset'].magnitude,
                omega.magnitude) for omega in self.omegas]

            self.test_results['sigma'][sigma.magnitude]['mu'][mu.magnitude] = {
                'absolute_value': np.abs(transfer_function),
                'phase': np.angle(transfer_function)}

    def test_frequencies(self):
        # take examplary frequencies for fixed sigma and mu
        sigma = self.network_params['sigma_1'].magnitude
        mu = self.network_params['mean_input_1'][0].magnitude
        ground_truth_data = self.ground_truth_result['sigma'][sigma]['mu'][mu]

        assert_array_equal(ground_truth_data['frequencies'],
                           self.frequencies.magnitude)

    def test_sigma_1_mu_1(self):
        # define specific sigma and mu
        sigma = self.network_params['sigma_1'].magnitude
        mu = self.network_params['mean_input_1'][0].magnitude
        ground_truth_data = self.ground_truth_result['sigma'][sigma]['mu'][mu]
        test_data = self.test_results['sigma'][sigma]['mu'][mu]

        print(self.frequencies[-1])
        assert_array_equal(ground_truth_data['absolute_value'],
                           test_data['absolute_value'])


    def test_sigma_1_mu_2(self):
        # define specific sigma and mu
        sigma = self.network_params['sigma_1'].magnitude
        mu = self.network_params['mean_input_1'][1].magnitude
        ground_truth_data = self.ground_truth_result['sigma'][sigma]['mu'][mu]
        test_data = self.test_results['sigma'][sigma]['mu'][mu]

        assert_array_equal(ground_truth_data['absolute_value'],
                           test_data['absolute_value'])

    def test_sigma_2_mu_1(self):
        # define specific sigma and mu
        sigma = self.network_params['sigma_2'].magnitude
        mu = self.network_params['mean_input_2'][0].magnitude
        # print(self.ground_truth_result['sigma'][sigma]['mu'].keys(), self.test_results['sigma'][sigma]['mu'].keys())
        ground_truth_data = self.ground_truth_result['sigma'][sigma]['mu'][mu]
        test_data = self.test_results['sigma'][sigma]['mu'][mu]


        assert_array_equal(ground_truth_data['absolute_value'],
                          test_data['absolute_value'])

    def test_sigma_2_mu_2(self):
        # define specific sigma and mu
        sigma = self.network_params['sigma_2'].magnitude
        mu = self.network_params['mean_input_2'][1].magnitude
        ground_truth_data = self.ground_truth_result['sigma'][sigma]['mu'][mu]
        test_data = self.test_results['sigma'][sigma]['mu'][mu]

        assert_array_equal(ground_truth_data['absolute_value'],
                           test_data['absolute_value'])





if __name__ == "__main__":
    unittest.main()
