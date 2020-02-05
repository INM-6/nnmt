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
from numpy.testing import assert_array_equal,assert_array_almost_equal, assert_allclose

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
            self.test_results['sigma'][sigma.magnitude]['mu'] = defaultdict(dict)
            for mu in self.network_params[f'mean_input_{index}']:

                # Stationary firing rates for delta shaped PSCs.
                nu_0 = lmt.aux_calcs.nu_0(
                    self.network_params['tau_m'],
                    self.network_params['tau_r'],
                    self.network_params['theta'],
                    self.network_params['V_reset'],
                    mu,
                    sigma)

                # Stationary firing rates for filtered synapses (via Taylor)
                nu0_fb = lmt.aux_calcs.nu0_fb(
                    self.network_params['tau_m'],
                    self.network_params['tau_s'],
                    self.network_params['tau_r'],
                    self.network_params['theta'],
                    self.network_params['V_reset'],
                    mu,
                    sigma)

                # Stationary firing rates for exp PSCs. (via shift)
                nu0_fb433 = lmt.aux_calcs.nu0_fb433(
                    self.network_params['tau_m'],
                    self.network_params['tau_s'],
                    self.network_params['tau_r'],
                    self.network_params['theta'],
                    self.network_params['V_reset'],
                    mu,
                    sigma)

                # colored noise zero-frequency limit of transfer function
                transfer_function_zero_freq = lmt.aux_calcs.d_nu_d_mu_fb433(
                    self.network_params['tau_m'],
                    self.network_params['tau_s'],
                    self.network_params['tau_r'],
                    self.network_params['theta'],
                    self.network_params['V_reset'],
                    mu,
                    sigma)

                transfer_function = lmt.meanfield_calcs.transfer_function(
                    [mu],
                    [sigma],
                    self.network_params['tau_m'],
                    self.network_params['tau_s'],
                    self.network_params['tau_r'],
                    self.network_params['theta'],
                    self.network_params['V_reset'],
                    self.network_params['dimension'],
                    self.omegas)

                self.test_results['sigma'][sigma.magnitude]['mu'][mu.magnitude] = {
                    'absolute_value': np.abs(transfer_function.magnitude.flatten()),
                    'phase': np.angle(transfer_function.magnitude.flatten()) / 2 / np.pi * 360,
                    'zero_freq': transfer_function_zero_freq.to(ureg.Hz/ureg.mV).magnitude,
                    'nu_0': nu_0.to(ureg.Hz).magnitude,
                    'nu0_fb': nu0_fb.to(ureg.Hz).magnitude,
                    'nu0_fb433': nu0_fb433.to(ureg.Hz).magnitude}


    #TODO in assert function first argument should be test data
    def test_frequencies(self):
        # take examplary frequencies for fixed sigma and mu
        sigma = self.network_params['sigma_1'].magnitude
        mu = self.network_params['mean_input_1'][0].magnitude
        ground_truth_data = self.ground_truth_result['sigma'][sigma]['mu'][mu]

        assert_array_equal(ground_truth_data['frequencies'],
                           self.frequencies.magnitude)

    def test_absolute_value(self):
        # define specific sigma and mu
        for index in [1,2]:
            sigma = self.network_params[f'sigma_{index}'].magnitude
            for mu in self.network_params[f'mean_input_{index}'].magnitude:
                print(sigma, mu)

                ground_truth_data = self.ground_truth_result['sigma'][sigma]['mu'][mu]
                test_data = self.test_results['sigma'][sigma]['mu'][mu]

                print('absolute_value')
                print(f'below {self.frequencies[100]}')
                assert_array_almost_equal(ground_truth_data['absolute_value'][:100],
                                          test_data['absolute_value'][:100],
                                          decimal = 4)

                print(f'below {self.frequencies[300]}')
                assert_array_almost_equal(ground_truth_data['absolute_value'][100:300],
                                          test_data['absolute_value'][100:300],
                                          decimal = 1)

                print(f'below {self.frequencies[-1]}')
                assert_allclose(ground_truth_data['absolute_value'],
                                          test_data['absolute_value'], atol=2)

    def test_phase(self):
        # define specific sigma and mu
        for index in [1,2]:
            sigma = self.network_params[f'sigma_{index}'].magnitude
            for mu in self.network_params[f'mean_input_{index}'].magnitude:
                print(sigma, mu)

                ground_truth_data = self.ground_truth_result['sigma'][sigma]['mu'][mu]
                test_data = self.test_results['sigma'][sigma]['mu'][mu]

                print('phase')
                print(f'below {self.frequencies[100]}')
                assert_allclose(ground_truth_data['phase'][:100],
                                          test_data['phase'][:100], atol=2)

                print(f'below {self.frequencies[300]}')
                assert_allclose(ground_truth_data['phase'][100:300],
                                          test_data['phase'][100:300], atol=50)

                print(f'below {self.frequencies[-1]}')
                assert_allclose(ground_truth_data['phase'],
                                          test_data['phase'], atol=150)

    def test_stationary_firing_rates(self):
        # define specific sigma and mu
        for index in [1,2]:
            sigma = self.network_params[f'sigma_{index}'].magnitude
            for mu in self.network_params[f'mean_input_{index}'].magnitude:
                print(sigma, mu)

                ground_truth_data = self.ground_truth_result['sigma'][sigma]['mu'][mu]
                test_data = self.test_results['sigma'][sigma]['mu'][mu]

                for key in test_data.keys():
                            if not key in ['absolute_value', 'phase']:
                                print(key)
                                assert_allclose(ground_truth_data[key],
                                                   test_data[key], atol=1e-12)




if __name__ == "__main__":
    unittest.main()
