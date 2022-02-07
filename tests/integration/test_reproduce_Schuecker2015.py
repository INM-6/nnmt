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

import nnmt.input_output as io

import nnmt
ureg = nnmt.ureg


config_path = 'tests/fixtures/integration/config/'
fix_path = 'tests/fixtures/integration/data/'

indices = [1, 2]


@pytest.fixture(scope='class')
def ground_truth_result():
    results = io.load_h5(fix_path + 'Schuecker2015_data.h5')
    return results


@pytest.fixture(scope='class')
def network_params():
    params = nnmt.input_output.load_val_unit_dict_from_yaml(
        config_path + 'Schuecker2015_parameters.yaml')
    params['dimension'] = 1
    nnmt.utils._strip_units(params)
    return params


@pytest.fixture(scope='class')
def si_network_params():
    params = nnmt.input_output.load_val_unit_dict_from_yaml(
        config_path + 'Schuecker2015_parameters.yaml')
    params['dimension'] = 1
    nnmt.utils._to_si_units(params)
    nnmt.utils._strip_units(params)
    return params


@pytest.fixture(scope='class')
def frequencies(network_params):
    frequencies = np.logspace(
        network_params['f_start_exponent'],
        network_params['f_end_exponent'],
        network_params['n_freqs'])
    return frequencies


@pytest.fixture(scope='class')
def omegas(frequencies):
    omegas = 2 * np.pi * frequencies
    return omegas


@pytest.fixture(scope='class')
def pre_results(si_network_params, omegas):
    # calculate nnmt results for different mus and sigmas
    absolute_values = []
    phases = []
    zero_freqs = []
    nu_0s = []
    nu0_fbs = []
    nu0_fb433s = []
    for i, index in enumerate(indices):
        # Stationary firing rates for delta shaped PSCs.
        nu_0 = nnmt.lif.delta._firing_rates_for_given_input(
            si_network_params[f'mean_input_{index}'],
            si_network_params[f'sigma_{index}'],
            si_network_params['V_reset'],
            si_network_params['theta'],
            si_network_params['tau_m'],
            si_network_params['tau_r'])

        # Stationary firing rates for filtered synapses (via shift)
        nu0_fb = nnmt.lif.exp._firing_rate_shift(
            si_network_params[f'mean_input_{index}'],
            si_network_params[f'sigma_{index}'],
            si_network_params['V_reset'],
            si_network_params['theta'],
            si_network_params['tau_m'],
            si_network_params['tau_r'],
            si_network_params['tau_s'])

        # Stationary firing rates for exp PSCs. (via Taylor)
        nu0_fb433 = nnmt.lif.exp._firing_rate_taylor(
            si_network_params[f'mean_input_{index}'],
            si_network_params[f'sigma_{index}'],
            si_network_params['V_reset'],
            si_network_params['theta'],
            si_network_params['tau_m'],
            si_network_params['tau_r'],
            si_network_params['tau_s'])

        # colored noise zero-frequency limit of transfer function
        transfer_function_zero_freq = (
            nnmt.lif.exp._derivative_of_firing_rates_wrt_mean_input(
                si_network_params[f'mean_input_{index}'],
                si_network_params[f'sigma_{index}'],
                si_network_params['V_reset'],
                si_network_params['theta'],
                si_network_params['tau_m'],
                si_network_params['tau_r'],
                si_network_params['tau_s'])) / 1000

        transfer_function = nnmt.lif.exp._transfer_function_shift(
            si_network_params[f'mean_input_{index}'],
            si_network_params[f'sigma_{index}'],
            si_network_params['tau_m'],
            si_network_params['tau_s'],
            si_network_params['tau_r'],
            si_network_params['theta'],
            si_network_params['V_reset'],
            omegas,
            synaptic_filter=False) / 1000

        # calculate properties plotted in Schuecker 2015
        absolute_value = np.abs(transfer_function)
        phase = (np.angle(transfer_function)
                 / 2 / np.pi * 360)
        zero_freq = transfer_function_zero_freq

        # collect all results
        absolute_values.append(absolute_value)
        phases.append(phase)
        zero_freqs.append(zero_freq)
        nu_0s.append(nu_0)
        nu0_fbs.append(nu0_fb)
        nu0_fb433s.append(nu0_fb433)

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
        test_results['sigma'][str(sigma)]['mu'] = (
            defaultdict(dict))
        for j, mu in enumerate(network_params[f'mean_input_{index}']):
            test_results[
                'sigma'][str(sigma)][
                'mu'][str(mu)] = {
                    'absolute_value': pre_results['absolute_values'][i][:, j],
                    'phase': pre_results['phases'][i][:, j],
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
        sigma = network_params['sigma_{}'.format(index)]
        mu = network_params['mean_input_{}'.format(index)][0]
        ground_truth_data = ground_truth_result['sigma'][str(sigma)]['mu'][str(mu)]
        assert_array_equal(frequencies,
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
        sigma = network_params['sigma_{}'.format(index)]
        for mu in network_params['mean_input_{}'.format(index)]:
            ground_truth_data = ground_truth_result['sigma'][str(sigma)]['mu'][str(mu)]
            test_data = test_result['sigma'][str(sigma)]['mu'][str(mu)]
            assert_allclose(test_data[key],
                            ground_truth_data[key], atol=1e-14)
