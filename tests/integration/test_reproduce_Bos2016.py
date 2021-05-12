# -*- coding: utf-8 -*-
"""
Integration tests reproducing data of the following
publication:

Bos, H., Diesmann, M. & Helias, M.
Identifying Anatomical Origins of Coexisting Oscillations in the Cortical
Microcircuit. PLoS Comput. Biol. 12, 1–34 (2016).
"""

import pytest
from collections import defaultdict

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_array_almost_equal
    )

import lif_meanfield_tools as lmt
from lif_meanfield_tools import ureg

import h5py_wrapper.wrapper as h5


config_path = 'tests/fixtures/integration/config/'
fix_path = 'tests/fixtures/integration/data/'

# options for debugging
save_data = False
use_saved_data = False


@pytest.fixture(scope='class')
def ground_truth_result():
    result = h5.load(fix_path + 'Bos2016_publicated_and_converted_data.h5')
    return result


@pytest.fixture(scope='class')
def bos_code_result():
    data = h5.load(fix_path + 'Bos2016_data.h5')
    return data


@pytest.fixture(scope='class')
def exemplary_frequency_idx(bos_code_result):
    return bos_code_result['exemplary_frequency_idx']


@pytest.fixture(scope='class')
def network(exemplary_frequency_idx):
    if use_saved_data:
        network = lmt.networks.Microcircuit(file=fix_path + 'network.h5')
    else:
        network = lmt.networks.Microcircuit(
            config_path + 'Bos2016_network_params.yaml',
            config_path + 'Bos2016_analysis_params.yaml')
    network.network_params['D'] = (
        lmt.networks.utils.delay_dist_matrix(
            network.network_params['Delay'],
            network.network_params['Delay_sd'],
            network.network_params['delay_dist'],
            network.analysis_params['omegas'])
        )
    
    lmt.lif.exp.working_point(network, method='taylor')
    lmt.lif.exp.transfer_function(network, method='taylor')
    lmt.lif.exp.effective_connectivity(network)
        
    omega = network.analysis_params['omegas'][exemplary_frequency_idx]
    network.analysis_params['omega'] = omega
    
    yield network
    
    if save_data:
        network.save(file=fix_path + 'network.h5', overwrite=True)


@pytest.fixture
def network_params(network):
    network._add_units_to_param_dicts_and_convert_to_input_units()
    params = network.network_params.copy()
    network._convert_param_dicts_to_base_units_and_strip_units()
    return params
    

@pytest.fixture
def bos_params(bos_code_result):
    params = bos_code_result['params'].copy()
    return params
    

@pytest.fixture(scope='class')
def freqs(network):
    fs = network.analysis_params['omegas'] / 2. / np.pi
    return fs


class Test_lif_meanfield_toolbox_vs_Bos_2016:

    @pytest.mark.parametrize('lmt_key, bos_key', [['populations',
                                                   'populations'],
                                                  ['N', 'N'],
                                                  ['C', 'C'],
                                                  ['tau_m', 'taum'],
                                                  ['tau_r', 'taur'],
                                                  ['tau_s', 'tauf'],
                                                  ['V_th_abs', 'Vth'],
                                                  ['V_0_abs', 'V0'],
                                                  ['d_e', 'de'],
                                                  ['d_i', 'di'],
                                                  ['d_e_sd', 'de_sd'],
                                                  ['d_i_sd', 'di_sd'],
                                                  ['delay_dist', 'delay_dist'],
                                                  ['w', 'w'],
                                                  ['K', 'I'],
                                                  ['g', 'g'],
                                                  ['nu_ext', 'v_ext'],
                                                  ['K_ext', 'Next'],
                                                  ['Delay', 'Delay'],
                                                  ['Delay_sd', 'Delay_sd'],
                                                  ])
    def test_network_parameters(self, network_params, bos_params,
                                lmt_key, bos_key):
        network_param = network_params[lmt_key]
        bos_param = bos_params[bos_key]
        if lmt_key == 'w':
            bos_param *= 2
        if isinstance(network_param, ureg.Quantity):
            network_param = network_param.magnitude
        network_params = np.atleast_1d(network_params)
        bos_param = np.atleast_1d(bos_param)
        try:
            assert_allclose(network_param, bos_param)
        except TypeError:
            assert_array_equal(network_param, bos_param)
            
    def test_analysis_frequencies(self, ground_truth_result, bos_code_result,
                                  freqs, exemplary_frequency_idx):
        ground_truth_data = ground_truth_result['fig_microcircuit']['freq_ana']
        bos_code_data = bos_code_result['omegas'] / 2. / np.pi
        test_data = freqs
        # check ground truth data vs data generated via old code
        assert_array_equal(bos_code_data, ground_truth_data)
        # check ground truth data vs data generated via lmt
        assert_array_equal(test_data, ground_truth_data)
        # check that the exemplary frequency is correct
        assert (test_data[exemplary_frequency_idx]
                == bos_code_data[exemplary_frequency_idx])

    def test_firing_rates(self, network, ground_truth_result, bos_code_result):
        ground_truth_data = ground_truth_result[
            'fig_microcircuit']['rates_calc']
        bos_code_data = bos_code_result['firing_rates']
        test_data = lmt.lif.exp.firing_rates(network, method='taylor'
                                             ).magnitude
        # check ground truth data vs data generated via old code
        assert_array_almost_equal(bos_code_data, ground_truth_data, decimal=5)
        # check ground truth data vs data generated via lmt
        assert_array_almost_equal(test_data, ground_truth_data, decimal=3)
        
    def test_delay_distribution_at_single_frequency(self, network,
                                                    bos_code_result):
        # ground truth data does not exist, but as regenerated bos_code_data
        # passes all comparisons to ground truth data, this can be assumed to
        # be fine
        bos_code_data = bos_code_result['delay_dist']
        omega = network.analysis_params['omega']
        test_data = lmt.networks.utils.delay_dist_matrix(
            network.network_params['Delay'],
            network.network_params['Delay_sd'],
            network.network_params['delay_dist'],
            [omega])[0]
        assert_array_equal(test_data.shape, bos_code_data.shape)
        assert_array_equal(test_data, bos_code_data)
        
    def test_effective_connectivity_at_single_frequency(self,
                                                        network,
                                                        bos_code_result):
        # ground truth data does not exist, but as regenerated bos_code_data
        # passes all comparisons to ground truth data, this can be assumed to
        # be fine
        bos_code_data = bos_code_result['MH']
        omega = network.analysis_params['omega']
        freqs = np.array([omega / np.pi / 2])
        lmt.lif.exp.transfer_function(network, freqs=freqs, method='taylor')
        D_old = network.network_params['D']
        network.network_params['D'] = lmt.networks.utils.delay_dist_matrix(
            network.network_params['Delay'],
            network.network_params['Delay_sd'],
            network.network_params['delay_dist'],
            [omega])
        test_data = lmt.lif.exp.effective_connectivity(network)[0]
        assert_array_almost_equal(test_data, bos_code_data, decimal=4)
        network.network_params['D'] = D_old

    def test_transfer_function(self, network, bos_code_result):
        # ground truth data does not exist, but as regenerated bos_code_data
        # passes all comparisons to ground truth data, this can be assumed to
        # be fine
        bos_code_data = bos_code_result[
            'transfer_function_with_synaptic_filter']
        # transfer functions are stored transposed
        test_data = lmt.lif.exp.transfer_function(network, method='taylor').T
        assert_array_equal(test_data.shape, bos_code_data.shape)
        assert_array_almost_equal(test_data / 1000, bos_code_data, decimal=4)
        
    def test_power_spectra(self, network, ground_truth_result,
                           bos_code_result):
        # Bos code actually calculates square of the power
        ground_truth_data = np.sqrt(
            ground_truth_result['fig_microcircuit']['power_ana'])
        bos_code_data = np.sqrt(bos_code_result['power_spectra'])
        # test regenerated data via original Bos code with publicated data
        assert_array_almost_equal(bos_code_data, ground_truth_data, decimal=3)
        # Bos code used Taylor method and the fortran implementation of the
        # Kummer's function to approximate the parabolic cylinder functions
        lmt.lif.exp.effective_connectivity(network)
        test_data = lmt.lif.exp.power_spectra(network).T
        assert_array_equal(test_data.shape, ground_truth_data.shape)
        assert_array_almost_equal(test_data, ground_truth_data, decimal=3)
    
    def test_eigenvalue_trajectories(self, network, ground_truth_result,
                                     bos_code_result):
        eigenvalue_spectra = eigenvalue_spectra = np.linalg.eig(
            lmt.lif.exp.effective_connectivity(network))[0].T
        ground_truth_data = ground_truth_result[
            'eigenvalue_trajectories']['eigs']
        bos_code_data = bos_code_result['eigenvalue_spectra']
        test_data = eigenvalue_spectra
        # need to bring the to the same shape (probably just calculated up to
        # 400 Hz in Bos paper)
        assert_array_almost_equal(
            bos_code_data.transpose()[:ground_truth_data.shape[0], :],
            ground_truth_data, decimal=4)
        # need to bring the to the same shape (probably just calculated up to
        # 400 Hz in Bos paper)
        assert_array_almost_equal(
            test_data.transpose()[:ground_truth_data.shape[0], :],
            ground_truth_data, decimal=4)
        
    def test_sensitivity_measure(self, network, ground_truth_result, freqs):
        power_spectra = lmt.lif.exp.power_spectra(network).T
        eigenvalue_spectra = np.linalg.eig(
            lmt.lif.exp.effective_connectivity(network))[0].T
        ground_truth_data = ground_truth_result['sensitivity_measure']
        
        # check whether highest power is identified correctly
        # TODO maybe not necessary
        pop_idx, freq_idx = np.unravel_index(np.argmax(power_spectra),
                                             np.shape(power_spectra))
        fmax = freqs[freq_idx]
        assert fmax == ground_truth_data['high_gamma1']['f_peak']

        # identify frequency which is closest to the point complex(1,0) per
        # eigenvalue trajectory
        sensitivity_dict = defaultdict(int)
        for eig_index, eig in enumerate(eigenvalue_spectra):
            critical_frequency = freqs[np.argmin(abs(eig - 1.0))]
            critical_frequency_index = np.argmin(
                abs(freqs - critical_frequency))
            critical_eigenvalue = eig[critical_frequency_index]

            sensitivity_dict[eig_index] = {
                'critical_frequency': critical_frequency,
                'critical_frequency_index': critical_frequency_index,
                'critical_eigenvalue': critical_eigenvalue}
        
        # identify which eigenvalue contributes most to this peak
        for eig_index, eig_results in sensitivity_dict.items():
            if eig_results['critical_frequency'] == fmax:
                eigenvalue_index = eig_index
                print(f"eigenvalue that contributes most to identified peak "
                      f"at {fmax}: {eigenvalue_index}")

        # see lines 224 ff in make_Bos2016_data/sensitivity_measure.py
        assert eigenvalue_index == 1

        # test sensitivity measure
        omegas = np.array([fmax * 2 * np.pi])
        network.analysis_params['omegas'] = omegas
        network.network_params['D'] = lmt.networks.utils.delay_dist_matrix(
            network.network_params['Delay'],
            network.network_params['Delay_sd'],
            network.network_params['delay_dist'],
            omegas)
        lmt.lif.exp.transfer_function(network, method='taylor')
        lmt.lif.exp.effective_connectivity(network)
        Z = lmt.lif.exp.sensitivity_measure(network)[0]
        assert_array_almost_equal(Z,
                                  ground_truth_data['high_gamma1']['Z'],
                                  decimal=4)

        # test critical eigenvalue
        eigc = sensitivity_dict[eigenvalue_index]['critical_eigenvalue']
        fmax = sensitivity_dict[eigenvalue_index]['critical_frequency']
        assert_array_almost_equal(eigc,
                                  ground_truth_data['high_gamma1']['eigc'],
                                  decimal=5)

        # test vector (k) between critical eigenvalue and complex(1,0)
        # and repective perpendicular (k_per)
        k = np.asarray([1, 0]) - np.asarray([eigc.real,
                                             eigc.imag])
        k /= np.sqrt(np.dot(k, k))
        k_per = np.asarray([-k[1], k[0]])
        k_per /= np.sqrt(np.dot(k_per, k_per))
        assert_array_almost_equal(k,
                                  ground_truth_data['high_gamma1']['k'],
                                  decimal=4)
        assert_array_almost_equal(k_per,
                                  ground_truth_data['high_gamma1']['k_per'],
                                  decimal=4)

        # test projections of sensitivty measure onto k and k_per
        Z_amp = Z.real * k[0] + Z.imag * k[1]
        Z_freq = Z.real * k_per[0] + Z.imag * k_per[1]
        assert_array_almost_equal(Z_amp,
                                  ground_truth_data['high_gamma1']['Z_amp'],
                                  decimal=4)
        assert_array_almost_equal(Z_freq,
                                  ground_truth_data['high_gamma1']['Z_freq'],
                                  decimal=4)
