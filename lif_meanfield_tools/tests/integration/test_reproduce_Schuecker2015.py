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
from numpy.testing import (assert_array_equal,
                           assert_allclose)

import lif_meanfield_tools as lmt
from ... import ureg

import h5py_wrapper.wrapper as h5

# TODO remove plotting after debugging
import matplotlib.pyplot as plt


class Test_Schuecker_2015_vs_lif_meanfield_toolbox(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Parameters to tweak the behavior of the test
        cls.plot_comparison = False

        # Load ground truth data
        cls.path_to_fixtures = ('./lif_meanfield_tools/tests/integration/'
                                'fixtures/')
        cls.ground_truth_result = h5.load(cls.path_to_fixtures
                                          + 'Schuecker2015_data.h5')

        # Load parameters for creation of test data
        cls.network_params = lmt.input_output.load_params(
            cls.path_to_fixtures + 'Schuecker2015_parameters.yaml')

        # Generate frequencies
        cls.frequencies = np.logspace(
            cls.network_params['f_start_exponent']['val'],
            cls.network_params['f_end_exponent']['val'],
            cls.network_params['n_freqs']['val']) * ureg.Hz
        cls.omegas = 2 * np.pi * cls.frequencies
        
        cls.network_params['dimension'] = 1

        cls.indices = [1, 2]
        
        # calculate lif_meanfield_tools results for different mus and sigmas
        cls.absolute_values = [[] for i in range(len(cls.indices))]
        cls.phases = [[] for i in range(len(cls.indices))]
        cls.zero_freqs = [[] for i in range(len(cls.indices))]
        cls.nu_0s = [[] for i in range(len(cls.indices))]
        cls.nu0_fbs = [[] for i in range(len(cls.indices))]
        cls.nu0_fb433s = [[] for i in range(len(cls.indices))]
        for i, index in enumerate(cls.indices):
            sigma = cls.network_params[f'sigma_{index}']
            for mu in cls.network_params[f'mean_input_{index}']:
                # Stationary firing rates for delta shaped PSCs.
                nu_0 = lmt.aux_calcs.nu_0(
                    cls.network_params['tau_m'],
                    cls.network_params['tau_r'],
                    cls.network_params['theta'],
                    cls.network_params['V_reset'],
                    mu,
                    sigma)

                # Stationary firing rates for filtered synapses (via Taylor)
                nu0_fb = lmt.aux_calcs.nu0_fb(
                    cls.network_params['tau_m'],
                    cls.network_params['tau_s'],
                    cls.network_params['tau_r'],
                    cls.network_params['theta'],
                    cls.network_params['V_reset'],
                    mu,
                    sigma)

                # Stationary firing rates for exp PSCs. (via shift)
                nu0_fb433 = lmt.aux_calcs.nu0_fb433(
                    cls.network_params['tau_m'],
                    cls.network_params['tau_s'],
                    cls.network_params['tau_r'],
                    cls.network_params['theta'],
                    cls.network_params['V_reset'],
                    mu,
                    sigma)

                # colored noise zero-frequency limit of transfer function
                transfer_function_zero_freq = lmt.aux_calcs.d_nu_d_mu_fb433(
                    cls.network_params['tau_m'],
                    cls.network_params['tau_s'],
                    cls.network_params['tau_r'],
                    cls.network_params['theta'],
                    cls.network_params['V_reset'],
                    mu,
                    sigma)

                transfer_function = lmt.meanfield_calcs.transfer_function(
                    [mu],
                    [sigma],
                    cls.network_params['tau_m'],
                    cls.network_params['tau_s'],
                    cls.network_params['tau_r'],
                    cls.network_params['theta'],
                    cls.network_params['V_reset'],
                    cls.network_params['dimension'],
                    cls.omegas,
                    synaptic_filter=False)
                
                # calculate properties plotted in Schuecker 2015
                absolute_value = np.abs(transfer_function.magnitude.flatten())
                phase = (np.angle(transfer_function.magnitude.flatten())
                         / 2 / np.pi * 360)
                zero_freq = (transfer_function_zero_freq.to(ureg.Hz / ureg.mV)
                             ).magnitude,
                nu_0 = nu_0.to(ureg.Hz).magnitude
                nu0_fb = nu0_fb.to(ureg.Hz).magnitude
                nu0_fb433.to(ureg.Hz).magnitude
                
                # collect all results
                cls.absolute_values[i].append(absolute_value)
                cls.phases[i].append(phase)
                cls.zero_freqs[i].append(zero_freq)
                cls.nu_0s[i].append(nu_0)
                cls.nu0_fbs[i].append(nu0_fb)
                cls.nu0_fb433s[i].append(nu0_fb433)
    
    def setUp(self):
        # Loop over the different values for sigma and mu and save results
        # in a dictionary analogously to ground_truth_data
        self.test_results = defaultdict(str)
        self.test_results['sigma'] = defaultdict(dict)
        for i, index in enumerate(self.indices):
            sigma = self.network_params[f'sigma_{index}']
            self.test_results['sigma'][sigma.magnitude]['mu'] = (
                defaultdict(dict))
            for j, mu in enumerate(self.network_params[f'mean_input_{index}']):
                self.test_results[
                    'sigma'][sigma.magnitude][
                    'mu'][mu.magnitude] = {
                        'absolute_value': self.absolute_values[i][j],
                        'phase': self.phases[i][j],
                        'zero_freq': self.zero_freqs[i][j],
                        'nu_0': self.nu_0s[i][j],
                        'nu0_fb': self.nu0_fbs[i][j],
                        'nu0_fb433': self.nu0_fb433s[i][j]}

    def test_frequencies_used_for_comparison_are_equal(self):
        for index in self.indices:
            with self.subTest(index=index):
                sigma = self.network_params['sigma_{}'.format(index)].magnitude
                mu = self.network_params[
                    'mean_input_{}'.format(index)][0].magnitude
                ground_truth_data = self.ground_truth_result[
                    'sigma'][sigma]['mu'][mu]
                assert_array_equal(self.frequencies.magnitude,
                                   ground_truth_data['frequencies'])

    def test_frequencies_used_for_comparison_2_are_equal(self):
        sigma = self.network_params['sigma_2'].magnitude
        mu = self.network_params['mean_input_2'][0].magnitude
        ground_truth_data = self.ground_truth_result['sigma'][sigma]['mu'][mu]
        assert_array_equal(self.frequencies.magnitude,
                           ground_truth_data['frequencies'])

    def test_absolute_value(self):
        # define specific sigma and mu
        for index in self.indices:
            with self.subTest(index=index):
                sigma = self.network_params[f'sigma_{index}'].magnitude
                for mu in self.network_params[f'mean_input_{index}'].magnitude:

                    ground_truth_data = (
                        self.ground_truth_result['sigma'][sigma]['mu'][mu])
                    test_data = self.test_results['sigma'][sigma]['mu'][mu]

                    assert_allclose(test_data['absolute_value'],
                                    ground_truth_data['absolute_value'],
                                    atol=1e-14)
            
                    if self.plot_comparison:
                        # plot for debugging - compare with
                        # fixtures/make_Schuecker_Fig4/PRE_Schuecker_Fig4.pdf
                        plt.title(f'$\mu$ = {mu}, $\sigma$ = {sigma}')
                        plt.semilogx(self.frequencies,
                                     ground_truth_data['absolute_value'],
                                     label='ground truth')

                        plt.semilogx(self.frequencies,
                                     test_data['absolute_value'], ls='--',
                                     label='test data')
                        plt.xlabel(r'frequency $\omega/2\pi\quad(1/'
                                   '\mathrm{s})$')
                        plt.ylabel(r'$|\frac{n(\omega)\nu}{\epsilon\mu}|\quad('
                                   '\mathrm{s}\,\mathrm{mV})^{-1}$',
                                   labelpad=0)
                        plt.legend()
                        plt.show()

    def test_phase(self):
        # define specific sigma and mu
        for index in self.indices:
            sigma = self.network_params[f'sigma_{index}'].magnitude
            for mu in self.network_params[f'mean_input_{index}'].magnitude:
                print(sigma, mu)

                ground_truth_data = (
                    self.ground_truth_result['sigma'][sigma]['mu'][mu])
                test_data = self.test_results['sigma'][sigma]['mu'][mu]

                print('phase')
                assert_allclose(test_data['phase'],
                                ground_truth_data['phase'],
                                atol=1e-14)

                if self.plot_comparison:
                    # plot for debugging - compare with
                    # fixtures/make_Schuecker_Fig4/PRE_Schuecker_Fig4.pdf
                    plt.title(f'$\mu$ = {mu}, $\sigma$ = {sigma}')
                    plt.semilogx(self.frequencies,
                                 ground_truth_data['phase'],
                                 label='ground truth')

                    plt.semilogx(self.frequencies,
                                 test_data['phase'], ls='--',
                                 label='test data')
                    plt.xlabel(r'frequency $\omega/2\pi\quad(1/\mathrm{s})$')
                    plt.ylabel(r'$-\angle n(\omega)\quad(^{\circ})$',
                               labelpad=2)
                    plt.legend()
                    plt.show()

    def test_stationary_firing_rates(self):
        # define specific sigma and mu
        for index in self.indices:
            sigma = self.network_params[f'sigma_{index}'].magnitude
            for mu in self.network_params[f'mean_input_{index}'].magnitude:
                print(sigma, mu)

                ground_truth_data = (
                    self.ground_truth_result['sigma'][sigma]['mu'][mu])
                test_data = self.test_results['sigma'][sigma]['mu'][mu]

                for key in ['nu_0', 'nu0_fb', 'nu0_fb433']:
                    print(key)
                    assert_allclose(test_data[key],
                                    ground_truth_data[key],
                                    atol=1e-14)

    def test_zero_frequency_limit(self):
        # define specific sigma and mu
        for index in self.indices:
            sigma = self.network_params[f'sigma_{index}'].magnitude
            for mu in self.network_params[f'mean_input_{index}'].magnitude:
                print(sigma, mu)

                ground_truth_data = (
                    self.ground_truth_result['sigma'][sigma]['mu'][mu])
                test_data = self.test_results['sigma'][sigma]['mu'][mu]

                key = 'zero_freq'
                print(key)
                assert_allclose(test_data[key],
                                ground_truth_data[key],
                                atol=1e-14)


if __name__ == "__main__":
    unittest.main()
