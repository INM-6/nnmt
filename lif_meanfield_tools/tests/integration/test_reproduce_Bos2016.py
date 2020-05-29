# -*- coding: utf-8 -*-
"""
Integration tests reproducing data of the following
publication:

Bos, H., Diesmann, M. & Helias, M.
Identifying Anatomical Origins of Coexisting Oscillations in the Cortical
Microcircuit. PLoS Comput. Biol. 12, 1–34 (2016).
"""

import unittest
import pytest
from collections import defaultdict

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

import lif_meanfield_tools as lmt
from ... import ureg

import h5py_wrapper.wrapper as h5

# TODO remove plotting after debugging
import matplotlib.pyplot as plt


fix_path = './lif_meanfield_tools/tests/integration/fixtures/'


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
    network = lmt.Network(fix_path + 'Bos2016_network_params.yaml',
                          fix_path + 'Bos2016_analysis_params.yaml')
    omega = network.analysis_params['omegas'][exemplary_frequency_idx]
    network.analysis_params['omega'] = omega
    return network


@pytest.fixture
def network_params(network):
    params = network.network_params.copy()
    return params
    

@pytest.fixture
def bos_params(bos_code_result):
    params = bos_code_result['params'].copy()
    return params
    

@pytest.fixture(scope='class')
def freqs(network):
    fs = network.analysis_params['omegas'].to(ureg.Hz).magnitude / 2. / np.pi
    return fs


@pytest.fixture(scope='class')
def firing_rates(network):
    rates = network.firing_rates()
    return rates


@pytest.fixture(scope='class')
def delay_dist(network, bos_data):
    omega = network.analysis_params['omega']
    delay_dist = network.delay_dist_matrix(omega)
    return delay_dist


@pytest.fixture(scope='class')
def transfer_function(network):
    return network.transfer_function()


@pytest.mark.parametrize('lmt_key, bos_key', [['populations', 'populations'],
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
def test_network_parameters(network_params, bos_params,
                            lmt_key, bos_key):
    network_param = network_params[lmt_key]
    bos_param = bos_params[bos_key]
    if lmt_key == 'w':
        bos_param *= 2
    if isinstance(network_param, ureg.Quantity):
        network_param = network_param.magnitude
    try:
        assert network_param == bos_param
    except ValueError:
        assert_array_equal(network_param, bos_param)
        
        
def test_analysis_frequencies(ground_truth_result, bos_code_result, freqs,
                              exemplary_frequency_idx):
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
    

def test_firing_rates(network, ground_truth_result, bos_code_result):
    ground_truth_data = ground_truth_result['fig_microcircuit']['rates_calc']
    bos_code_data = bos_code_result['firing_rates']
    test_data = network.firing_rates().to(ureg.Hz).magnitude
    # check ground truth data vs data generated via old code
    assert_array_almost_equal(bos_code_data, ground_truth_data, decimal=5)
    # check ground truth data vs data generated via lmt
    assert_array_almost_equal(test_data, ground_truth_data, decimal=5)
    
    
@pytest.mark.select
def test_delay_distribution_at_single_frequency(network, bos_code_result):
    # ground truth data does not exist, but as regenerated bos_code_data
    # passes all comparisons to ground truth data, this can be assumed to
    # be fine
    bos_code_data = bos_code_result['delay_dist']
    omega = network.analysis_params['omega']
    test_data = network.delay_dist_matrix_single(omega)
    assert_array_equal(test_data.shape, bos_code_data.shape)
    assert_array_equal(test_data.magnitude, bos_code_data)



class BosTestCase(unittest.TestCase):
    
    def setUp(self):
        # Parameters to tweak the behavior of the test
        self.save_data = False
        self.use_saved_data = False
        self.plot_comparison = False

        # Load ground truth data
        self.path_to_fixtures = ('./lif_meanfield_tools/tests/integration/'
                                 'fixtures/')
        self.ground_truth_result = h5.load(
            self.path_to_fixtures + 'Bos2016_publicated_and_converted_data.h5')
        self.bos_code_result = h5.load(
            self.path_to_fixtures + 'Bos2016_data.h5')

        self.exemplary_frequency_idx = self.bos_code_result[
            'exemplary_frequency_idx']

        self.network_params = (
            self.path_to_fixtures + 'Bos2016_network_params.yaml')
        self.analysis_params = (
            self.path_to_fixtures + 'Bos2016_analysis_params.yaml')

        self.network = lmt.Network(network_params=self.network_params,
                                   analysis_params=self.analysis_params)

        self.freqs = (self.network.analysis_params['omegas']
                      ).to(ureg.Hz).magnitude / 2. / np.pi

        # In the case that test have to run several times for debugging
        # it can be advantageous to once save intermediate results and load
        # those in subsequent runs

        # TODO raise error if files not found
        if self.use_saved_data:
            try:
                self.transfer_functions = (np.load(
                    self.path_to_fixtures + 'Bos_test_transfer_function.npy')
                    ) * ureg.Hz / ureg.mV
            except FileNotFoundError:
                print('Need to run one of the following tests with save_data '
                      'being set to True:',
                      'test_transfer_function',
                      'test_effective_connectivity_at_single_frequency')

            try:
                self.power_spectra = (np.load(
                    self.path_to_fixtures + 'Bos_test_power_spectra.npy')
                    ) * ureg.Hz
            except FileNotFoundError:
                print('Need to run one of the following tests with save_data '
                      'being set to True:',
                      'test_power_spectra', 'test_sensitivity_measure')

            try:
                self.eigenvalue_spectra = np.load(
                    self.path_to_fixtures + 'Bos_test_eigenvalue_spectra.npy')
            except FileNotFoundError:
                print('Need to run one of the following tests with save_data '
                      'being set to True:',
                      'test_eigenvalue_trajectories',
                      'test_sensitivity_measure')

    def test_network_parameters(self):
        bos_code_data = self.bos_code_result['params']
        test_data = self.network.network_params
        assert_array_equal(test_data['populations'],
                           bos_code_data['populations'])
        # number of neurons in populations
        assert_array_equal(test_data['N'], bos_code_data['N'])
        # membrane capacitance
        self.assertEqual(test_data['C'].magnitude, bos_code_data['C'])
        # membrane time constant
        self.assertEqual(test_data['tau_m'].magnitude, bos_code_data['taum'])
        # refractory time
        self.assertEqual(test_data['tau_r'].magnitude, bos_code_data['taur'])
        # absolute reset potential
        self.assertEqual(test_data['V_0_abs'].magnitude, bos_code_data['V0'])
        # absolute threshold of membrane potential
        self.assertEqual(test_data['V_th_abs'].magnitude, bos_code_data['Vth'])

        # synaptic time constant
        self.assertEqual(test_data['tau_s'].magnitude, bos_code_data['tauf'])
        # delay of excitatory connections
        self.assertEqual(test_data['d_e'].magnitude, bos_code_data['de'])
        # delay of inhibitory connections
        self.assertEqual(test_data['d_i'].magnitude, bos_code_data['di'])
        # standard deviation of delay of excitatory connections
        self.assertEqual(test_data['d_e_sd'].magnitude, bos_code_data['de_sd'])
        # standard deviation of delay of inhibitory connections
        self.assertEqual(test_data['d_i_sd'].magnitude, bos_code_data['di_sd'])
        # delay distribution
        self.assertEqual(test_data['delay_dist'], bos_code_data['delay_dist'])
        # PSC amplitude
        # TODO factor of 2
        self.assertEqual(test_data['w'].magnitude, 2 * bos_code_data['w'])
        # indegrees
        assert_array_equal(test_data['K'], bos_code_data['I'])
        # ratio of inhibitory to excitatory weights
        self.assertEqual(test_data['g'], bos_code_data['g'])
        # firing rate of external input
        self.assertEqual(test_data['nu_ext'].magnitude, bos_code_data['v_ext'])
        # number of external neurons
        assert_array_equal(test_data['K_ext'], bos_code_data['Next'])

        assert_array_equal(test_data['Delay'].magnitude,
                           bos_code_data['Delay'])

        assert_array_equal(test_data['Delay_sd'].magnitude,
                           bos_code_data['Delay_sd'])

    def test_analysis_frequencies(self):
        ground_truth_data = self.ground_truth_result[
            'fig_microcircuit']['freq_ana']
        bos_code_data = self.bos_code_result['omegas'] / 2. / np.pi
        test_data = self.freqs

        # check ground truth data vs data generated via old code
        assert_array_equal(bos_code_data, ground_truth_data)

        # check ground truth data vs data generated via lmt
        assert_array_equal(test_data, ground_truth_data)

        # check that the exemplary frequency is correct
        self.assertEqual(test_data[self.exemplary_frequency_idx],
                         bos_code_data[self.exemplary_frequency_idx])

    def test_firing_rates(self):
        ground_truth_data = self.ground_truth_result[
            'fig_microcircuit']['rates_calc']
        bos_code_data = self.bos_code_result['firing_rates']
        test_data = self.network.firing_rates().to(ureg.Hz).magnitude

        # check ground truth data vs data generated via old code
        assert_array_almost_equal(bos_code_data, ground_truth_data, decimal=5)
        # check ground truth data vs data generated via lmt
        assert_array_almost_equal(test_data, ground_truth_data, decimal=5)

    def test_delay_distribution_at_single_frequency(self):
        # ground truth data does not exist, but as regenerated bos_code_data
        # passes all comparisons to ground truth data, this can be assumed to
        # be fine
        bos_code_data = self.bos_code_result['delay_dist']
        omega = self.network.analysis_params[
            'omegas'][self.exemplary_frequency_idx]
        test_data = self.network.delay_dist_matrix_single(omega)

        assert_array_equal(test_data.shape, bos_code_data.shape)
        assert_array_equal(test_data.magnitude, bos_code_data)

    def test_effective_connectivity_at_single_frequency(self):
        # ground truth data does not exist, but as regenerated bos_code_data
        # passes all comparisons to ground truth data, this can be assumed to
        # be fine
        bos_code_data = self.bos_code_result['MH']
        omega = self.network.analysis_params[
            'omegas'][self.exemplary_frequency_idx]

        if not self.use_saved_data:
            self.transfer_functions = self.network.transfer_function(
                method='taylor')

        test_data = lmt.meanfield_calcs._effective_connectivity(
            omega,
            self.transfer_functions[self.exemplary_frequency_idx],
            self.network.network_params['tau_m'],
            self.network.network_params['J'],
            self.network.network_params['K'],
            self.network.network_params['dimension'],
            self.network.delay_dist_matrix_single(omega))

        assert_array_almost_equal(test_data.to_base_units(), bos_code_data,
                                  decimal=5)

        if self.save_data:
            np.save(self.path_to_fixtures + 'Bos_test_transfer_function.npy',
                    self.transfer_functions.magnitude)

    def test_transfer_function(self):
        if not self.use_saved_data:
            self.transfer_functions = self.network.transfer_function(
                method='taylor')

        # ground truth data does not exist, but as regenerated bos_code_data
        # passes all comparisons to ground truth data, this can be assumed to
        # be fine
        bos_code_data = self.bos_code_result[
            'transfer_function_with_synaptic_filter']
        test_data = self.transfer_functions.transpose()

        assert_array_equal(test_data.shape, bos_code_data.shape)
        # Transfer functions are stored transposed
        assert_array_almost_equal(test_data, bos_code_data, decimal=4)

        if self.plot_comparison:
            populations = self.network.network_params['populations']
            fig = plt.figure()
            for i, trans_func in enumerate(bos_code_data):
                plt.plot(self.freqs, trans_func,
                         label=f'bos: {populations[i]}')

            for i, trans_func in enumerate(test_data):
                plt.plot(self.freqs, trans_func, ls='--',
                         label=f'lmt: {populations[i]}')
            plt.legend()
            plt.title('Transfer Functions')
            plt.show()

        if self.save_data:
            np.save(self.path_to_fixtures + 'Bos_test_transfer_function.npy',
                    self.transfer_functions.magnitude)

    def test_power_spectra(self):
        if not self.use_saved_data:
            self.power_spectra = self.network.power_spectra(method='taylor')

        # Bos code actually calculates square of the power
        ground_truth_data = np.sqrt(
            self.ground_truth_result['fig_microcircuit']['power_ana'])
        bos_code_data = np.sqrt(self.bos_code_result['power_spectra'])

        # test regenerated data via original Bos code with publicated data
        assert_array_almost_equal(bos_code_data, ground_truth_data, decimal=3)

        # Bos code used Taylor method and the fortran implementation of the
        # Kummer's function to approximate the parabolic cylinder functions
        test_data = self.power_spectra

        assert_array_equal(test_data.shape, ground_truth_data.shape)
        assert_array_almost_equal(test_data, ground_truth_data, decimal=3)
        if self.plot_comparison:
            # TODO improve this plot
            populations = self.network.network_params['populations']
            nx = 5
            ny = 4

            fig = plt.figure()

            fig.suptitle('Power Spectra - Meanfield', y=0.5)
            fig.subplots_adjust(wspace=0.1, hspace=0.2, top=0.93,
                                bottom=0.185, left=0.1, right=0.97)
            ax = [plt.subplot2grid((nx, ny), (nx - (2 - i % 2),
                                              int(np.floor(i / 2))))
                  for i in range(8)]  # spectra

            for layer in [0, 1, 2, 3]:
                for pop in [0, 1]:
                    j = layer * 2 + pop
                    box = ax[j].get_position()
                    ax[j].set_position([box.x0, box.y0 - box.height * 0.58,
                                        box.width, box.height])
                    ax[j].plot(self.freqs, ground_truth_data[j],
                               label='ground_truth_data', zorder=1)
                    ax[j].plot(self.freqs, bos_code_data[j], ls='--',
                               label='bos_code_data', zorder=2)
                    ax[j].plot(self.freqs, test_data[j], ls=(0, (3, 5, 1, 5)),
                               label='test', zorder=3)
                    ax[j].set_xlim([-5.0, 400.0])
                    ax[j].set_ylim([1e-6, 1e1])
                    ax[j].set_yscale('log')
                    ax[j].set_yticks([])
                    ax[j].set_xticks([100, 200, 300])
                    ax[j].set_title(populations[j])
                    ax[j].legend()
                    if pop == 0:
                        ax[j].set_xticklabels([])
                    else:
                        box = ax[j].get_position()
                        ax[j].set_position([box.x0, box.y0 - box.height * 0.2,
                                            box.width, box.height])
                        ax[j].set_xlabel(r'frequency (1/$s$)', fontsize=12)
                ax[0].set_yticks([1e-5, 1e-3, 1e-1])
                ax[1].set_yticks([1e-5, 1e-3, 1e-1])
                ax[0].set_ylabel(r'$|C(\omega)|$')
                ax[1].set_ylabel(r'$|C(\omega)|$')
            fig.tight_layout()
            plt.show()
            
        if self.save_data:
            np.save(self.path_to_fixtures + 'Bos_test_power_spectra.npy',
                    self.power_spectra.magnitude)

    def test_eigenvalue_trajectories(self):
        if not self.use_saved_data:
            self.eigenvalue_spectra = self.network.eigenvalue_spectra(
                'MH', method='taylor')

        ground_truth_data = self.ground_truth_result[
            'eigenvalue_trajectories']['eigs']
        bos_code_data = self.bos_code_result['eigenvalue_spectra']
        test_data = self.eigenvalue_spectra

        # need to bring the to the same shape (probably just calculated up to
        # 400 Hz in Bos paper)
        assert_array_almost_equal(
            bos_code_data.transpose()[:ground_truth_data.shape[0], :],
            ground_truth_data, decimal=4)

        print('eigenvalue_trajectories are stored transposed and just '
              'calculate up to 400Hz')
        print(test_data.shape)
        print(ground_truth_data.shape)

        # need to bring the to the same shape (probably just calculated up to
        # 400 Hz in Bos paper)
        assert_array_almost_equal(
            test_data.transpose()[:ground_truth_data.shape[0], :],
            ground_truth_data, decimal=4)

        # identify frequency which is closest to the point complex(1,0) per
        # eigenvalue trajectory
        self.sensitivity_dict = defaultdict(int)
        for eig_index, eig in enumerate(self.eigenvalue_spectra):
            critical_frequency = self.freqs[np.argmin(abs(eig - 1.0))]
            critical_frequency_index = np.argmin(
                abs(self.freqs - critical_frequency))
            critical_eigenvalue = eig[critical_frequency_index]

            self.sensitivity_dict[eig_index] = {
                'critical_frequency': critical_frequency,
                'critical_frequency_index': critical_frequency_index,
                'critical_eigenvalue': critical_eigenvalue}

        if self.plot_comparison:
            # Eigenvalue Trajectories
            fig = plt.figure()
            fig.suptitle('Eigenvalue Trajectories')
            for eig_index, eig_results in self.sensitivity_dict.items():
                sc = plt.scatter(np.real(test_data[eig_index]),
                                 np.imag(test_data[eig_index]),
                                 c=self.freqs, cmap='viridis', s=0.5)
                plt.scatter(np.real(eig_results['critical_eigenvalue']),
                            np.imag(eig_results['critical_eigenvalue']),
                            marker='+',
                            s=30, label=eig_index)
            plt.legend(title='Eigenvalue Index')
            plt.xlabel('Re($\lambda(\omega)$)')
            plt.ylabel('Im($\lambda(\omega)$)')
            plt.scatter(1, 0, marker='*', s=15, color='black')
            plt.ylim([-4, 6.5])
            plt.xlim([-11.5, 2])
            cbar = plt.colorbar(sc)
            cbar.set_label('Frequency $\omega$ [Hz]')
            plt.clim(np.min(self.freqs), np.max(self.freqs))
            plt.show()

            # Eigenvalue Trajectories - Zoomed In
            fig = plt.figure()
            fig.suptitle('Eigenvalue Trajectories Zoom')
            for eig_index, eig_results in self.sensitivity_dict.items():
                sc = plt.scatter(np.real(test_data[eig_index]),
                                 np.imag(test_data[eig_index]),
                                 c=self.freqs, cmap='viridis', s=0.5)
                plt.scatter(np.real(eig_results['critical_eigenvalue']),
                            np.imag(eig_results['critical_eigenvalue']),
                            marker='+',
                            s=30, label=eig_index)
            plt.legend(title='Eigenvalue Index')
            plt.xlabel('Re($\lambda(\omega)$)')
            plt.ylabel('Im($\lambda(\omega)$)')
            plt.scatter(1, 0, marker='*', s=15, color='black')
            plt.ylim([-0.2, 0.5])
            plt.xlim([-0.5, 2])
            cbar = plt.colorbar(sc)
            cbar.set_label('Frequency $\omega$ [Hz]')
            plt.clim(np.min(self.freqs), np.max(self.freqs))
            plt.show()

        if self.save_data:
            np.save(self.path_to_fixtures + 'Bos_test_eigenvalue_spectra.npy',
                    self.eigenvalue_spectra.magnitude)

    def test_sensitivity_measure(self):
        if not self.use_saved_data:
            self.power_spectra = self.network.power_spectra(method='taylor')
            self.eigenvalue_spectra = self.network.eigenvalue_spectra(
                'MH', method='taylor')

        ground_truth_data = self.ground_truth_result['sensitivity_measure']

        # check whether highest power is identified correctly
        # TODO maybe not necessary
        pop_idx, freq_idx = np.unravel_index(np.argmax(self.power_spectra),
                                             np.shape(self.power_spectra))
        fmax = self.freqs[freq_idx]
        self.assertEqual(fmax, ground_truth_data['high_gamma1']['f_peak'])

        # identify frequency which is closest to the point complex(1,0) per
        # eigenvalue trajectory
        self.sensitivity_dict = defaultdict(int)
        for eig_index, eig in enumerate(self.eigenvalue_spectra):
            critical_frequency = self.freqs[np.argmin(abs(eig - 1.0))]
            critical_frequency_index = np.argmin(
                abs(self.freqs - critical_frequency))
            critical_eigenvalue = eig[critical_frequency_index]

            self.sensitivity_dict[eig_index] = {
                'critical_frequency': critical_frequency,
                'critical_frequency_index': critical_frequency_index,
                'critical_eigenvalue': critical_eigenvalue}

        # identify which eigenvalue contributes most to this peak
        for eig_index, eig_results in self.sensitivity_dict.items():
            if eig_results['critical_frequency'] == fmax:
                eigenvalue_index = eig_index
                print(f'eigenvalue that contributes most to identified peak at'
                      ' {fmax}: {eigenvalue_index}')

        # see lines 224 ff in make_Bos2016_data/sensitivity_measure.py
        self.assertEqual(eigenvalue_index, 1)

        # test sensitivity measure
        Z = self.network.sensitivity_measure(fmax * ureg.Hz, method='taylor')
        assert_array_almost_equal(Z,
                                  ground_truth_data['high_gamma1']['Z'],
                                  decimal=4)

        # test critical eigenvalue
        eigc = self.sensitivity_dict[eigenvalue_index]['critical_eigenvalue']
        fmax = self.sensitivity_dict[eigenvalue_index]['critical_frequency']
        assert_array_almost_equal(eigc,
                                  ground_truth_data['high_gamma1']['eigc'],
                                  decimal=5)

        # test vector (k) between critical eigenvalue and complex(1,0)
        # and repective perpendicular (k_per)
        k = np.asarray([1, 0]) - np.asarray([eigc.real, eigc.imag])
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

        if self.plot_comparison:
            populations = self.network.network_params['populations']

            # identify colorbar limits
            zmin = np.min(np.real(Z))
            zmax = np.max(np.real(Z))
            z = np.max([abs(zmin), abs(zmax)])

            # real(Z)
            fig, ax = plt.subplots()
            ax.set_title('$\\mathcal{R}(Z($' + str(np.round(fmax, 1)) + '))')
            heatmap = ax.imshow(np.real(Z), vmin=-z, vmax=z,
                                cmap=plt.cm.coolwarm)
            fig.colorbar(heatmap, fraction=0.046, pad=0.04)
            ax.set_xticks(np.arange(len(populations)))
            ax.set_yticks(np.arange(len(populations)))
            ax.set_xticklabels(populations)
            ax.set_yticklabels(populations)
            plt.tight_layout()
            plt.show()

            # imag(Z)
            fig, ax = plt.subplots()
            ax.set_title('$\\mathcal{I}(Z($' + str(np.round(fmax, 1)) + '))')
            heatmap = ax.imshow(np.imag(Z), vmin=-z, vmax=z,
                                cmap=plt.cm.coolwarm)
            fig.colorbar(heatmap, fraction=0.046, pad=0.04)
            ax.set_xticks(np.arange(len(populations)))
            ax.set_yticks(np.arange(len(populations)))
            ax.set_xticklabels(populations)
            ax.set_yticklabels(populations)
            plt.tight_layout()
            plt.show()

            # Z_amp
            fig, ax = plt.subplots()
            heatmap = ax.imshow(Z_amp, vmin=-z, vmax=z, cmap=plt.cm.coolwarm)
            ax.set_title('$(Z_{amp}($' + str(np.round(fmax, 1)) + '))')
            fig.colorbar(heatmap, fraction=0.046, pad=0.04)
            ax.set_xticks(np.arange(len(populations)))
            ax.set_yticks(np.arange(len(populations)))
            ax.set_xticklabels(populations)
            ax.set_yticklabels(populations)
            plt.tight_layout()
            plt.show()

            # Z_freq
            fig, ax = plt.subplots()
            heatmap = ax.imshow(Z_freq, vmin=-z, vmax=z, cmap=plt.cm.coolwarm)
            ax.set_title('$(Z_{freq}($' + str(np.round(fmax, 1)) + '))')
            fig.colorbar(heatmap, fraction=0.046, pad=0.04)
            ax.set_xticks(np.arange(len(populations)))
            ax.set_yticks(np.arange(len(populations)))
            ax.set_xticklabels(populations)
            ax.set_yticklabels(populations)
            plt.tight_layout()
            plt.show()

        if self.save_data:
            np.save(self.path_to_fixtures + 'Bos_test_eigenvalue_spectra.npy',
                    self.eigenvalue_spectra.magnitude)
            np.save(self.path_to_fixtures + 'Bos_test_power_spectra.npy',
                    self.power_spectra.magnitude)
