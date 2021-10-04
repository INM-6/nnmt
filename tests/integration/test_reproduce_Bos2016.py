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

import nnmt
from nnmt import ureg

import nnmt.input_output as io


config_path = 'tests/fixtures/integration/config/'
fix_path = 'tests/fixtures/integration/data/'

# options for debugging
save_data = False
use_saved_data = False


@pytest.fixture(scope='class')
def ground_truth_result():
    result = io.load_h5(fix_path + 'Bos2016_publicated_and_converted_data.h5')
    return result


@pytest.fixture(scope='class')
def bos_code_result():
    data = io.load_h5(fix_path + 'Bos2016_data.h5')
    return data


@pytest.fixture(scope='class')
def exemplary_frequency_idx(bos_code_result):
    return bos_code_result['exemplary_frequency_idx']


@pytest.fixture(scope='class')
def network(exemplary_frequency_idx):
    if use_saved_data:
        network = nnmt.models.Microcircuit(file=fix_path + 'network.h5')
    else:
        network = nnmt.models.Microcircuit(
            config_path + 'Bos2016_network_params.yaml',
            config_path + 'Bos2016_analysis_params.yaml')
    nnmt.lif.exp.working_point(network, method='taylor')
    nnmt.lif.exp.transfer_function(network, method='taylor')
    nnmt.network_properties.delay_dist_matrix(network)
    nnmt.lif.exp.effective_connectivity(network)
        
    omega = network.analysis_params['omegas'][exemplary_frequency_idx]
    network.analysis_params['omega'] = omega
    
    yield network.copy()
    
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

    @pytest.mark.parametrize('nnmt_key, bos_key', [['populations',
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
                                nnmt_key, bos_key):
        network_param = network_params[nnmt_key]
        bos_param = bos_params[bos_key]
        if nnmt_key == 'w':
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
        # check ground truth data vs data generated via nnmt
        assert_array_equal(test_data, ground_truth_data)
        # check that the exemplary frequency is correct
        assert (test_data[exemplary_frequency_idx]
                == bos_code_data[exemplary_frequency_idx])

    def test_firing_rates(self, network, ground_truth_result, bos_code_result):
        ground_truth_data = ground_truth_result[
            'fig_microcircuit']['rates_calc']
        bos_code_data = bos_code_result['firing_rates']
        test_data = nnmt.lif.exp.firing_rates(network, method='taylor')
        # check ground truth data vs data generated via old code
        assert_array_almost_equal(bos_code_data, ground_truth_data, decimal=5)
        # check ground truth data vs data generated via nnmt
        assert_array_almost_equal(test_data, ground_truth_data, decimal=3)
        
    def test_delay_distribution_at_single_frequency(self, network,
                                                    bos_code_result):
        # ground truth data does not exist, but as regenerated bos_code_data
        # passes all comparisons to ground truth data, this can be assumed to
        # be fine
        bos_code_data = bos_code_result['delay_dist']
        omega = network.analysis_params['omega']
        test_data = nnmt.network_properties.delay_dist_matrix(
            network, omega / 2 / np.pi)[0]
        assert_array_equal(test_data.shape, bos_code_data.shape)
        assert_array_equal(test_data, bos_code_data)
        nnmt.network_properties.delay_dist_matrix(network)
        
    def test_effective_connectivity_at_single_frequency(self,
                                                        network,
                                                        bos_code_result):
        # ground truth data does not exist, but as regenerated bos_code_data
        # passes all comparisons to ground truth data, this can be assumed to
        # be fine
        bos_code_data = bos_code_result['MH']
        omega = network.analysis_params['omega']
        freqs = np.array([omega / np.pi / 2])
        nnmt.lif.exp.transfer_function(network, freqs=freqs, method='taylor')
        nnmt.network_properties.delay_dist_matrix(network, omega / 2 / np.pi)
        test_data = nnmt.lif.exp.effective_connectivity(network)[0]
        assert_array_almost_equal(test_data, bos_code_data, decimal=4)
        nnmt.network_properties.delay_dist_matrix(network)

    def test_transfer_function(self, network, bos_code_result):
        # ground truth data does not exist, but as regenerated bos_code_data
        # passes all comparisons to ground truth data, this can be assumed to
        # be fine
        bos_code_data = bos_code_result[
            'transfer_function_with_synaptic_filter']
        # transfer functions are stored transposed
        test_data = nnmt.lif.exp.transfer_function(network, method='taylor').T
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
        nnmt.lif.exp.effective_connectivity(network)
        test_data = nnmt.lif.exp.power_spectra(network).T
        assert_array_equal(test_data.shape, ground_truth_data.shape)
        assert_array_almost_equal(test_data, ground_truth_data, decimal=3)
    
    def test_eigenvalue_trajectories(self, network, ground_truth_result,
                                     bos_code_result):
        eigenvalue_spectra = eigenvalue_spectra = np.linalg.eig(
            nnmt.lif.exp.effective_connectivity(network))[0].T
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
        ground_truth_data = ground_truth_result['sensitivity_measure']
                
        sensitivity_dict = nnmt.lif.exp.sensitivity_measure_per_eigenmode(
            network)
        
        # in the Bos results, the eigenvalue were not resorted to match
        # across frequencies, hence to compare the eigenvalue indices
        # the resorting has to be reversed
        eigenvalues = np.linalg.eig(
            nnmt.lif.exp.effective_connectivity(network))[0]
        _, resorting_mask = nnmt.lif.exp._resort_eigenvalues(eigenvalues)

        # loop through different eigenvalues and check corresponding results
        # from Bos
        for eig_index in range(network.network_params['dimension']):
            freq_idx = sensitivity_dict[eig_index]['critical_frequency_index']
            not_resorted_eig_index = resorting_mask[freq_idx][eig_index]
            if not_resorted_eig_index in [0, 1, 2, 3]:
                truth = ground_truth_data[f'high_gamma{not_resorted_eig_index}']
            elif not_resorted_eig_index == 4:
                truth = ground_truth_data[f'gamma{not_resorted_eig_index}']
            elif not_resorted_eig_index == 6:
                truth = ground_truth_data[f'low{not_resorted_eig_index}']
            else:
                print(f'{not_resorted_eig_index} not stored in ground truth')
                continue
                
            print(eig_index, not_resorted_eig_index)
            
            # check frequency which is closest to the point complex(1,0)    
            fmax = sensitivity_dict[eig_index]['critical_frequency']
            # allow for one frequency step deviation
            df = network.analysis_params['df']
            assert fmax-df <= truth['f_peak'] <= fmax+df

            # test sensitivity measure
            sensitivity = sensitivity_dict[eig_index]['sensitivity']
            assert_array_almost_equal(sensitivity,
                                    truth['Z'],
                                    decimal=2)

            # test critical eigenvalue
            eigc = sensitivity_dict[eig_index]['critical_eigenvalue']
            assert_array_almost_equal(eigc,
                                    truth['eigc'],
                                    decimal=3)

            # test vector (k) between critical eigenvalue and complex(1,0)
            # and repective perpendicular (k_per)
        
            k = sensitivity_dict[eig_index]['k']
            k_per = sensitivity_dict[eig_index]['k_per']
            
            assert_array_almost_equal(k,
                                    truth['k'],
                                    decimal=2)
            assert_array_almost_equal(k_per,
                                    truth['k_per'],
                                    decimal=2)

            # test projections of sensitivty measure onto k and k_per
            Z_amp = sensitivity_dict[eig_index]['sensitivity_amp']
            Z_freq = sensitivity_dict[eig_index]['sensitivity_freq']        
            
            assert_array_almost_equal(Z_amp,
                                    truth['Z_amp'],
                                    decimal=2)
            assert_array_almost_equal(Z_freq,
                                    truth['Z_freq'],
                                    decimal=2)
