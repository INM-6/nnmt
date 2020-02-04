# -*- coding: utf-8 -*-
"""
Integration tests reproducing data for the Figure 4 of the following
publication:

Schuecker, J., Diesmann, M. & Helias, M.
Modulated escape from a metastable state driven by colored noise.
Phys. Rev. E - Stat. Nonlinear, Soft Matter Phys. 92, 1–11 (2015).
"""

import unittest
from collections import defaultdict

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

import lif_meanfield_tools as lmt
from ... import ureg

import h5py_wrapper.wrapper as h5

class SchueckerTestCase(unittest.TestCase):
    def setUp(self):
        # Load ground truth data
        self.path_to_fixtures = './lif_meanfield_tools/tests/integration/fixtures/'
        self.ground_truth_result = h5.load(self.path_to_fixtures +
                                           'Schuecker2015_data.h5')

        # Generate test data
        self.network_params = lmt.input_output.load_params(
            self.path_to_fixtures + 'Schuecker2015_parameters.yaml')

        # Generate frequencies
        self.frequencies = np.logspace(
            self.network_params['f_start_exponent']['val'],
            self.network_params['f_end_exponent']['val'],
            self.network_params['n_freqs']['val']) * ureg.Hz

        self.omegas = 2 * np.pi * self.frequencies
        self.network_params['dimension'] = 1

        # Loop over the different values for sigma and mu and save results
        # in a dictionary analogously to ground_truth_data
        self.test_results = defaultdict(str)
        self.test_results['sigma'] = defaultdict(dict)

        for index in [1,2]:
            sigma = self.network_params[f'sigma_{index}']
            for mu in self.network_params[f'mean_input_{index}']:
                self.test_results['sigma'][sigma.magnitude]['mu'] = defaultdict(dict)

                transfer_function = lmt.meanfield_calcs.transfer_function(
                    [mu],
                    [sigma],
                    self.network_params['tau_m'],
                    self.network_params['tau_s'],
                    self.network_params['tau_r'],
                    self.network_params['theta'],
                    self.network_params['V_reset'],
                    self.network_params['dimension'],
                    self.omegas[:2])

                self.test_results['sigma'][sigma.magnitude]['mu'][mu.magnitude] = {
                    'absolute_value': np.abs(transfer_function),
                    'phase': np.angle(transfer_function.magnitude)}


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
