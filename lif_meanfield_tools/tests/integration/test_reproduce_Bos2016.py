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

class BosTestCase(unittest.TestCase):
    def setUp(self):
        # Load ground truth data
        self.path_to_fixtures = './lif_meanfield_tools/tests/integration/fixtures/'
        self.ground_truth_result = h5.load(self.path_to_fixtures +
                                           'Bos2016_data.h5')

        self.network_params = self.path_to_fixtures + 'Bos2016_network_params.yaml'
        self.analysis_params = self.path_to_fixtures + 'Bos2016_analysis_params.yaml'

        self.network = lmt.Network(network_params=self.network_params,
                              analysis_params=self.analysis_params)

        # For some tests it will be useful to define a limited range of frequencies
        # print(self.network.analysis_params['omegas'][200] /2./np.pi)
        self.low_omegas = self.network.analysis_params['omegas'][:100]
        # print(self.network.analysis_params['omegas'][500] /2./np.pi)
        self.moderate_omegas = self.network.analysis_params['omegas'][100:300]
        # print(self.network.analysis_params['omegas'] / 2./np.pi)
        self.high_omegas = self.network.analysis_params['omegas'][300:]

    def test_firing_rates(self):
        ground_truth_data = self.ground_truth_result['fig_microcircuit']['rates_calc']
        test_data = self.network.firing_rates().to(ureg.Hz).magnitude
        assert_array_almost_equal(test_data, ground_truth_data, decimal = 5)

    def test_analysis_frequencies(self):
        ground_truth_data = self.ground_truth_result['fig_microcircuit']['freq_ana']
        test_data = self.network.analysis_params['omegas'].to(ureg.Hz).magnitude /2./np.pi
        print(ground_truth_data.shape)
        print(test_data.shape)
        print(self.network.analysis_params.keys())
        assert_array_almost_equal(test_data, ground_truth_data)

    def test_power_spectra(self):
        # Bos code actually calculates square of the power
        ground_truth_data = np.sqrt(self.ground_truth_result['fig_microcircuit']['power_ana'])
        # Bos code used Taylor method and the fortran implementation of the Kummer's function
        # to approximate the parabolic cylinder functions
        self.network.transfer_function(method='taylor')
        test_data = self.network.power_spectra()

        print(ground_truth_data.shape)
        print(test_data.shape)
        print(self.network.analysis_params.keys())
        # The mismatch depends on the frequency!
        # up to 20 Hz
        idx1 = 126
        assert_array_almost_equal(test_data[:,:idx1], ground_truth_data[:,:idx1],
                                  decimal=3)
        # total
        assert_array_almost_equal(test_data, ground_truth_data,
                                  decimal=0)



        assert_array_almost_equal([1.2], [1.3], decimal=0)
        assert_array_almost_equal([1.2], [2.3], decimal=0)
        assert_array_almost_equal([1.2], [3.0], decimal=0)
        # maybe better work with allclose
