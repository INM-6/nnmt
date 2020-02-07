# -*- coding: utf-8 -*-
"""
Integration tests reproducing data of the following
publication:

Bos, H., Diesmann, M. & Helias, M.
Identifying Anatomical Origins of Coexisting Oscillations in the Cortical
Microcircuit. PLoS Comput. Biol. 12, 1–34 (2016).
"""

import unittest

import numpy as np
from numpy.testing import assert_array_equal,assert_array_almost_equal, assert_allclose

import lif_meanfield_tools as lmt
from ... import ureg

import h5py_wrapper.wrapper as h5

import matplotlib.pyplot as plt

class BosTestCase(unittest.TestCase):
    def setUp(self):
        # Load ground truth data
        self.path_to_fixtures = './lif_meanfield_tools/tests/integration/fixtures/'
        self.ground_truth_result = h5.load(self.path_to_fixtures +
                                           'Bos2016_publicated_and_converted_data.h5')
        self.bos_code_result = h5.load(self.path_to_fixtures +
                                           'Bos2016_data.h5')
        # TODO do this more nicely
        self.exemplary_frequency_idx = self.bos_code_result['exemplary_frequency_idx']

        self.network_params = self.path_to_fixtures + 'Bos2016_network_params.yaml'
        self.analysis_params = self.path_to_fixtures + 'Bos2016_analysis_params.yaml'

        self.network = lmt.Network(network_params=self.network_params,
                              analysis_params=self.analysis_params)
        self.transfer_functions = np.load(self.path_to_fixtures +
                                          'Bos_test_transfer_function.npy')
        self.power_spectra = np.load(self.path_to_fixtures +
                                          'Bos_test_power_spectra.npy')
        self.eigenvalue_spectra = np.load(self.path_to_fixtures +
                                          'Bos_test_eigenvalue_spectra.npy')


    def test_network_parameters(self):
        bos_code_data = self.bos_code_result['params']
        test_data = self.network.network_params

        assert_array_equal(test_data['populations'], bos_code_data['populations'])
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
        self.assertEqual(test_data['w'].magnitude, 2*bos_code_data['w'])
        # indegrees
        assert_array_equal(test_data['K'], bos_code_data['I'])
        # ratio of inhibitory to excitatory weights
        self.assertEqual(test_data['g'], bos_code_data['g'])
        # firing rate of external input
        self.assertEqual(test_data['nu_ext'].magnitude, bos_code_data['v_ext'])
        # number of external neurons
        assert_array_equal(test_data['K_ext'], bos_code_data['Next'])

        assert_array_equal(test_data['Delay'].magnitude, bos_code_data['Delay'])
        assert_array_equal(test_data['Delay_sd'].magnitude, bos_code_data['Delay_sd'])

    def test_analysis_frequencies(self):
        ground_truth_data = self.ground_truth_result['fig_microcircuit']['freq_ana']
        bos_code_data = self.bos_code_result['omegas'] /2./np.pi
        test_data = self.network.analysis_params['omegas'].to(ureg.Hz).magnitude /2./np.pi

        # check ground truth data vs data generated via old code
        assert_array_equal(bos_code_data, ground_truth_data)

        # check ground truth data vs data generated via lmt
        assert_array_equal(test_data, ground_truth_data)

        # check that the exemplary frequency is correct
        self.assertEqual(test_data[self.exemplary_frequency_idx],
                           bos_code_data[self.exemplary_frequency_idx])

    def test_firing_rates(self):
        ground_truth_data = self.ground_truth_result['fig_microcircuit']['rates_calc']
        bos_code_data = self.bos_code_result['firing_rates']
        test_data = self.network.firing_rates().to(ureg.Hz).magnitude

        # check ground truth data vs data generated via old code
        assert_array_almost_equal(bos_code_data, ground_truth_data, decimal = 5)

        # check ground truth data vs data generated via lmt
        assert_array_almost_equal(test_data, ground_truth_data, decimal = 5)

    def test_delay_distribution_at_single_frequency(self):
        bos_code_data = self.bos_code_result['delay_dist']
        omega = self.network.analysis_params['omegas'][self.exemplary_frequency_idx]
        test_data = self.network.delay_dist_matrix_single(omega)


        assert_array_equal(test_data.shape, bos_code_data.shape)
        assert_array_equal(test_data.magnitude, bos_code_data)

    def test_effective_connectivity_at_single_frequency(self):
        bos_code_data = self.bos_code_result['MH']
        omega = self.network.analysis_params['omegas'][self.exemplary_frequency_idx]
        test_data = lmt.meanfield_calcs._effective_connectivity(omega,
                                                         self.transfer_function[20],
                                                         self.network.network_params['tau_m'],
                                                         self.network.network_params['J'],
                                                         self.network.network_params['K'],
                                                         self.network.network_params['dimension'],
                                                         self.network.delay_dist_matrix_single(omega))


        print(test_data, bos_code_data)
        assert_array_equal(test_data, bos_code_data)


    def test_transfer_function(self):
        bos_code_data = self.bos_code_result['transfer_function_with_synaptic_filter']
        # test_data = self.network.transfer_function(method='taylor')
        test_data = self.transfer_functions

        print(bos_code_data.shape, test_data.shape)
        # plot for debugging
        freqs = self.network.analysis_params['omegas']/2./np.pi
        populations = self.network.network_params['populations']
        fig = plt.figure()
        for i, trans_func in enumerate(bos_code_data):
            plt.plot(freqs, trans_func, label=f'bos: {populations[i]}')

        for i, trans_func in enumerate(test_data.transpose()):
            plt.plot(freqs, trans_func, ls='--', label=f'lmt: {populations[i]}')
        plt.legend()
        plt.title('Transfer Functions')
        plt.show()

        # Transfer functions are stored transposed
        assert_array_almost_equal(test_data.transpose(), bos_code_data, decimal = 3)
        # # save output
        # np.save(self.path_to_fixtures + 'Bos_test_transfer_function.npy', test_data.magnitude)

    def test_power_spectra(self):
        # Bos code actually calculates square of the power
        ground_truth_data = np.sqrt(self.ground_truth_result['fig_microcircuit']['power_ana'])
        bos_code_data = np.sqrt(self.bos_code_result['power_spectra'])


        # Bos code used Taylor method and the fortran implementation of the Kummer's function
        # to approximate the parabolic cylinder functions
        # self.network.transfer_function(method='taylor')
        # test_data = self.network.power_spectra(method='taylor')
        test_data = self.power_spectra
        # save output
        # np.save(self.path_to_fixtures + 'Bos_test_power_spectra.npy', test_data.magnitude)


        print(ground_truth_data.shape)
        # print(test_data.shape)
        print(self.network.analysis_params.keys())

        assert_array_almost_equal(test_data, ground_truth_data,decimal=3)

        # plot for debugging
        # TODO improve this plot
        freqs = self.network.analysis_params['omegas']/2./np.pi
        populations = self.network.network_params['populations']
        nx = 5
        ny = 4

        fig = plt.figure()

        fig.suptitle('Power Spectra - Meanfield', y=0.5)
        fig.subplots_adjust(wspace=0.1, hspace=0.2, top=0.93,
                            bottom=0.185, left=0.1, right=0.97)
        ax = [plt.subplot2grid((nx,ny), (nx-(2-i%2),int(np.floor(i/2)))) for i in range(8)] # spectra

        for layer in [0, 1, 2, 3]:
            for pop in [0, 1]:
                j = layer*2+pop
                box = ax[j].get_position()
                ax[j].set_position([box.x0, box.y0-box.height*0.58,
                                       box.width, box.height])
                ax[j].plot(freqs, ground_truth_data[j],
                           label='ground_truth_data', zorder=1)
                ax[j].plot(freqs, bos_code_data[j], ls='--',
                           label='bos_code_data', zorder=2)
                ax[j].plot(freqs, test_data[j], ls=(0, (3, 5, 1, 5)),
                           label='test', zorder=3)
                ax[j].set_xlim([-5.0, 400.0])
                ax[j].set_ylim([1e-6, 1e1])
                ax[j].set_yscale('log')
                ax[j].set_yticks([])
                ax[j].set_xticks([100, 200, 300])
                ax[j].set_title(populations[j])
                # ax[j].legend()
                if pop == 0:
                    ax[j].set_xticklabels([])
                else:
                    box = ax[j].get_position()
                    ax[j].set_position([box.x0, box.y0-box.height*0.2,
                                           box.width, box.height])
                    ax[j].set_xlabel(r'frequency (1/$s$)', fontsize=12)
            ax[0].set_yticks([1e-5,1e-3,1e-1])
            ax[1].set_yticks([1e-5,1e-3,1e-1])
            ax[0].set_ylabel(r'$|C(\omega)|$')
            ax[1].set_ylabel(r'$|C(\omega)|$')
        fig.tight_layout()
        plt.show()



    def test_eigenvalue_trajectories(self):
        ground_truth_data = self.ground_truth_result['eigenvalue_trajectories']['eigs']
        # test_data = self.network.eigenvalue_spectra('MH', method='taylor')

        test_data = self.eigenvalue_spectra
        # save output
        # np.save(self.path_to_fixtures + 'Bos_test_eigenvalue_spectra.npy', test_data)

        print(test_data.shape)
        print(ground_truth_data.shape)

        # need to bring the to the same shape (probably just calculated up to 400 Hz in Bos paper)
        assert_array_almost_equal(test_data.transpose()[:ground_truth_data.shape[0],:], ground_truth_data, decimal=4)

        freqs = self.network.analysis_params['omegas'].to(ureg.Hz).magnitude /2./np.pi
        power = self.power_spectra
        eigs = self.eigenvalue_spectra
        populations = self.network.network_params['populations']

        # identify frequency with maximal power
        eig_index, freq_idx =  np.unravel_index(np.argmax(power), np.shape(power))
        fmax = freqs[freq_idx]
        print(fmax)

        eigcs = []
        critical_indices = []
        for eig_index, eig in enumerate(eigs):
            print(eig.shape)
            fmax = freqs[np.argmin(abs(eig-1.0))]
            print(fmax)
            fmax_index = np.argmin(abs(freqs-fmax))
            eigc = eig[fmax_index]
            print(eigc)
            eigcs.append(eigc)
            critical_indices.append((eig_index, fmax_index))



        fig = plt.figure()
        fig.suptitle('Eigenvalue Trajectories')
        for i in range(8):
            sc = plt.scatter(np.real(eigs[i]), np.imag(eigs[i]),
                        c=freqs, cmap='viridis',
                        s=0.5)
        # for i in range(8):
        #     plt.scatter(np.real(eigs[i][freq_idx]), np.imag(eigs[i][freq_idx]),
        #                 marker='p',
        #                 s=30, label=i)
        for i in range(8):
            plt.scatter(np.real(eigs[critical_indices[i][0]][critical_indices[i][1]]),
                        np.imag(eigs[critical_indices[i][0]][critical_indices[i][1]]),
                        marker='+',
                        s=30, label=i)

        plt.legend()
        plt.xlabel('Re($\lambda(\omega)$)')
        plt.ylabel('Im($\lambda(\omega)$)')
        plt.scatter(1,0, marker='*', s=15, color='black')
        plt.ylim([-4, 6.5])
        plt.xlim([-11.5, 2])
        cbar = plt.colorbar(sc)
        cbar.set_label('Frequency $\omega$ [Hz]')
        plt.clim(np.min(freqs), np.max(freqs))
        plt.show()

        fig = plt.figure()
        fig.suptitle('Eigenvalue Trajectories Zoom')
        for i in range(8):
            sc = plt.scatter(np.real(eigs[i]), np.imag(eigs[i]),
                        c=freqs, cmap='viridis',
                        s=0.5)
        for i in range(8):
            plt.scatter(np.real(eigs[i][freq_idx]), np.imag(eigs[i][freq_idx]),
                        marker='p',
                        s=30 , label=i)
        plt.legend()
        plt.xlabel('Re($\lambda(\omega)$)')
        plt.ylabel('Im($\lambda(\omega)$)')
        plt.scatter(1,0, marker='*', s=15, color='black')
        plt.ylim([-0.2, 0.5])
        plt.xlim([-0.5, 2])
        cbar = plt.colorbar(sc)
        cbar.set_label('Frequency $\omega$ [Hz]')
        plt.clim(np.min(freqs), np.max(freqs))
        plt.show()

        fig = plt.figure()
        fig.suptitle('Critical Eigenvalue')
        plt.scatter(freqs, np.real(eigs[eig_index]), label="real", s=1)
        plt.scatter(freqs, np.imag(eigs[eig_index]), label="imag", s=1)
        plt.scatter(freqs, abs(eigs[eig_index]-1), label="abs($\lambda$ - 1)", s=1)
        plt.axvline(x=fmax, ymin=-5, ymax=5, label=f'fmax={np.round(fmax,1)}Hz')
        plt.legend()
        plt.show()

    def test_sensitivity_measure(self):
        ground_truth_data = self.ground_truth_result['sensitivity_measure']


        freqs = self.network.analysis_params['omegas'].to(ureg.Hz).magnitude /2./np.pi
        power = self.power_spectra
        eigs = self.eigenvalue_spectra
        populations = self.network.network_params['populations']

        # Adapt to shapes of ground_truth_data
        # freqs = freqs[:ground_truth_data['freqs'].shape[0]]
        # eigs = eigs.transpose()[:ground_truth_data['eigs'].shape[0],:]

        # assert_array_almost_equal(freqs, ground_truth_data['freqs'])
        # assert_array_almost_equal(eigs, ground_truth_data['eigs'], decimal=4)


        pop_idx, freq_idx =  np.unravel_index(np.argmax(power), np.shape(power))
        freqs = self.network.analysis_params['omegas'].to(ureg.Hz).magnitude /2./np.pi
        fmax = freqs[freq_idx]
        self.assertEqual(fmax, ground_truth_data['high_gamma1']['f_peak'])

        print(pop_idx, fmax)



        # calculate sensitivity measure
        Z = self.network.sensitivity_measure(fmax*ureg.Hz, method='taylor')
        assert_array_almost_equal(Z, ground_truth_data['high_gamma1']['Z'], decimal=4)

        # eigc = eigs[pop_idx][np.argmin(abs(eigs[pop_idx]-1))]
        eigcs = []
        critical_indices = []
        for eig_index, eig in enumerate(eigs):
            print(eig.shape)
            fmax = freqs[np.argmin(abs(eig-1.0))]
            print(fmax)
            fmax_index = np.argmin(abs(freqs-fmax))
            eigc = eig[fmax_index]
            print(eigc)
            eigcs.append(eigc)
            critical_indices.append((eig_index, fmax_index))
        # TODO check why this is failing

        eigc = eigcs[1]
        fmax = freqs[critical_indices[1][1]]

        assert_array_almost_equal(eigc, ground_truth_data['high_gamma1']['eigc'], decimal=5)

        k = np.asarray([1, 0])-np.asarray([eigc.real, eigc.imag])
        k /= np.sqrt(np.dot(k, k))
        k_per = np.asarray([-k[1], k[0]])
        k_per /= np.sqrt(np.dot(k_per, k_per))
        assert_array_almost_equal(k, ground_truth_data['high_gamma1']['k'], decimal=4)
        assert_array_almost_equal(k_per, ground_truth_data['high_gamma1']['k_per'], decimal=4)

        Z_amp = Z.real*k[0]+Z.imag*k[1]
        Z_freq = Z.real*k_per[0]+Z.imag*k_per[1]
        assert_array_almost_equal(Z_amp, ground_truth_data['high_gamma1']['Z_amp'], decimal=4)
        assert_array_almost_equal(Z_freq, ground_truth_data['high_gamma1']['Z_freq'], decimal=4)


        zmin = np.min(np.real(Z))
        zmax = np.max(np.real(Z))
        z = np.max([abs(zmin), abs(zmax)])

        # plotting

        fig, ax = plt.subplots()
        ax.set_title('$\\mathcal{R}(Z($' + str(np.round(fmax,1)) +'))')
        heatmap = ax.imshow(np.real(Z), vmin=-z, vmax=z, cmap=plt.cm.coolwarm)
        fig.colorbar(heatmap, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(populations)))
        ax.set_yticks(np.arange(len(populations)))
        ax.set_xticklabels(populations)
        ax.set_yticklabels(populations)
        plt.tight_layout()
        plt.show()



        fig, ax = plt.subplots()
        ax.set_title('$\\mathcal{I}(Z($' + str(np.round(fmax,1)) +'))')
        heatmap = ax.imshow(np.imag(Z), vmin=-z, vmax=z, cmap=plt.cm.coolwarm)
        fig.colorbar(heatmap, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(populations)))
        ax.set_yticks(np.arange(len(populations)))
        ax.set_xticklabels(populations)
        ax.set_yticklabels(populations)
        plt.tight_layout()
        plt.show()


        fig, ax = plt.subplots()
        heatmap = ax.imshow(Z_amp, vmin=-z, vmax=z, cmap=plt.cm.coolwarm)
        ax.set_title('$(Z_{amp}($' + str(np.round(fmax,1)) +'))')
        fig.colorbar(heatmap, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(populations)))
        ax.set_yticks(np.arange(len(populations)))
        ax.set_xticklabels(populations)
        ax.set_yticklabels(populations)
        plt.tight_layout()
        plt.show()


        fig, ax = plt.subplots()
        heatmap = ax.imshow(Z_freq, vmin=-z, vmax=z, cmap=plt.cm.coolwarm)
        ax.set_title('$(Z_{freq}($' + str(np.round(fmax,1)) +'))')
        fig.colorbar(heatmap, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(populations)))
        ax.set_yticks(np.arange(len(populations)))
        ax.set_xticklabels(populations)
        ax.set_yticklabels(populations)
        plt.tight_layout()
        plt.show()
