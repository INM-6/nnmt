# -*- coding: utf-8 -*-
"""
Integration tests reproducing data for the Figure 4 of the following
publication:

Schuecker, J., Diesmann, M. & Helias, M.
Modulated escape from a metastable state driven by colored noise.
Phys. Rev. E - Stat. Nonlinear, Soft Matter Phys. 92, 1–11 (2015).
"""

import pytest
from collections import defaultdict

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

import lif_meanfield_tools as lmt
from lif_meanfield_tools import ureg

import h5py_wrapper.wrapper as h5


config_path = 'tests/fixtures/integration/config/'
fix_path = 'tests/fixtures/integration/data/'

indices = [1, 2]


@pytest.fixture(scope='class')
def ground_truth_result():
    results = h5.load(fix_path + 'Schuecker2015_data.h5')
    return results


@pytest.fixture(scope='class')
def network_params():
    params = lmt.input_output.load_params(
        config_path + 'Schuecker2015_parameters.yaml')
    params['dimension'] = 1
    return params


@pytest.fixture(scope='class')
def frequencies(network_params):
    frequencies = np.logspace(
        network_params['f_start_exponent']['val'],
        network_params['f_end_exponent']['val'],
        network_params['n_freqs']['val']) * ureg.Hz
    return frequencies
    
    
@pytest.fixture(scope='class')
def omegas(frequencies):
    omegas = 2 * np.pi * frequencies
    return omegas


@pytest.fixture(scope='class')
def pre_results(network_params, omegas):
    # calculate lif_meanfield_tools results for different mus and sigmas
    absolute_values = [[] for i in range(len(indices))]
    phases = [[] for i in range(len(indices))]
    zero_freqs = [[] for i in range(len(indices))]
    nu_0s = [[] for i in range(len(indices))]
    nu0_fbs = [[] for i in range(len(indices))]
    nu0_fb433s = [[] for i in range(len(indices))]
    for i, index in enumerate(indices):
        sigma = network_params[f'sigma_{index}']
        for mu in network_params[f'mean_input_{index}']:
            # Stationary firing rates for delta shaped PSCs.
            nu_0 = lmt.aux_calcs.nu_0(
                network_params['tau_m'],
                network_params['tau_r'],
                network_params['theta'],
                network_params['V_reset'],
                mu,
                sigma)

            # Stationary firing rates for filtered synapses (via Taylor)
            nu0_fb = lmt.aux_calcs.nu0_fb(
                network_params['tau_m'],
                network_params['tau_s'],
                network_params['tau_r'],
                network_params['theta'],
                network_params['V_reset'],
                mu,
                sigma)

            # Stationary firing rates for exp PSCs. (via shift)
            nu0_fb433 = lmt.aux_calcs.nu0_fb433(
                network_params['tau_m'],
                network_params['tau_s'],
                network_params['tau_r'],
                network_params['theta'],
                network_params['V_reset'],
                mu,
                sigma)

            # colored noise zero-frequency limit of transfer function
            transfer_function_zero_freq = lmt.aux_calcs.d_nu_d_mu_fb433(
                network_params['tau_m'],
                network_params['tau_s'],
                network_params['tau_r'],
                network_params['theta'],
                network_params['V_reset'],
                mu,
                sigma)

            transfer_function = lmt.meanfield_calcs.transfer_function(
                lmt.utils.pint_array([mu]),
                lmt.utils.pint_array([sigma]),
                network_params['tau_m'],
                network_params['tau_s'],
                network_params['tau_r'],
                network_params['theta'],
                network_params['V_reset'],
                network_params['dimension'],
                omegas,
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
            absolute_values[i].append(absolute_value)
            phases[i].append(phase)
            zero_freqs[i].append(zero_freq)
            nu_0s[i].append(nu_0)
            nu0_fbs[i].append(nu0_fb)
            nu0_fb433s[i].append(nu0_fb433)
    
    pre_results = dict(
        absolute_values=absolute_values,
        phases=phases,
        zero_freqs=zero_freqs,
        nu_0s=nu_0s,
        nu0_fbs=nu0_fbs,
        nu0_fb433s=nu0_fb433s)
    return pre_results
    

@pytest.fixture
def test_result(pre_results, network_params):
    # Loop over the different values for sigma and mu and save results
    # in a dictionary analogously to ground_truth_data
    test_results = defaultdict(str)
    test_results['sigma'] = defaultdict(dict)
    for i, index in enumerate(indices):
        sigma = network_params[f'sigma_{index}']
        test_results['sigma'][sigma.magnitude]['mu'] = (
            defaultdict(dict))
        for j, mu in enumerate(network_params[f'mean_input_{index}']):
            test_results[
                'sigma'][sigma.magnitude][
                'mu'][mu.magnitude] = {
                    'absolute_value': pre_results['absolute_values'][i][j],
                    'phase': pre_results['phases'][i][j],
                    'zero_freq': pre_results['zero_freqs'][i][j],
                    'nu_0': pre_results['nu_0s'][i][j],
                    'nu0_fb': pre_results['nu0_fbs'][i][j],
                    'nu0_fb433': pre_results['nu0_fb433s'][i][j]}
    return test_results


class Test_lif_meanfield_toolbox_vs_Schuecker_2015:
    
    @pytest.mark.parametrize('index', indices)
    def test_frequencies_used_for_comparison_are_equal(self, index,
                                                       network_params,
                                                       frequencies,
                                                       ground_truth_result):
        sigma = network_params['sigma_{}'.format(index)].magnitude
        mu = network_params['mean_input_{}'.format(index)][0].magnitude
        ground_truth_data = ground_truth_result['sigma'][sigma]['mu'][mu]
        assert_array_equal(frequencies.magnitude,
                           ground_truth_data['frequencies'])

    @pytest.mark.parametrize('key', ['absolute_value',
                                     'phase',
                                     'nu_0',
                                     'nu0_fb',
                                     'nu0_fb433',
                                     'zero_freq'])
    @pytest.mark.parametrize('index', indices)
    def test_results_coincide(self, key, index, network_params,
                              ground_truth_result, test_result):
        sigma = network_params['sigma_{}'.format(index)].magnitude
        for mu in network_params['mean_input_{}'.format(index)].magnitude:
            ground_truth_data = ground_truth_result['sigma'][sigma]['mu'][mu]
            test_data = test_result['sigma'][sigma]['mu'][mu]
            try:
                assert_allclose(test_data[key].magnitude,
                                ground_truth_data[key], atol=1e-14)
            except AttributeError:
                assert_allclose(test_data[key],
                                ground_truth_data[key], atol=1e-14)
