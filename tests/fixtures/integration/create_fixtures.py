#!/usr/bin/env python
# encoding:utf8
'''
Creates fixtures for nnmt integration tests.

WARNING: Only use this script, if your code is trustworthy! The script runs
         the nnmt code to produce the fixtures that are then
         stored in h5 format. If you run this script and your code is not
         working correctly, a lot of tests will pass despite your code giving
         wrong results.

If you still want to run this script type: python create_fixtures.py -f

Usage: create_fixtures.py [options]

Options:
    --all                create all integration fixtures
    -f, --force        force code to run
    --firing_rates_fully_vectorized
    -h, --help         show this information
'''

import docopt
import sys

import nnmt
import numpy as np


if __name__ == '__main__':
    # always show help message if not invoked with -f option
    if len(sys.argv) == 1:
        sys.argv.append('-h')

    args = docopt.docopt(__doc__)

    fixture_path = 'integration/data/'
    config_path = 'integration/config/'

    # only run code if users are sure they want to do it
    if args['--force']:

        network = nnmt.models.Microcircuit(
            config_path + 'network_params.yaml',
            config_path + 'analysis_params.yaml')

        omega = network.analysis_params['omega']
        frequency = omega/(2*np.pi)
        margin = network.analysis_params['margin']

        mean_input_set = network.network_params['mean_input_set']
        std_input_set = network.network_params['std_input_set']
        network.results[
            'lif.exp.firing_rates_taylor'] = nnmt.lif.exp.firing_rates(
            network, method='taylor')
        nnmt.lif.exp.working_point(network, method='shift')
        network.results[
            'delay_dist_matrix'] = nnmt.network_properties.delay_dist_matrix(
                network)
        network.results['lif.exp.tf_single'] = nnmt.lif.exp.transfer_function(
            network, omega)
        network.results['lif.exp.tf_taylor'] = nnmt.lif.exp.transfer_function(
            network, method='taylor')
        network.results['lif.exp.tf_shift'] = nnmt.lif.exp.transfer_function(
            network, method='shift')
        nnmt.lif.exp.effective_connectivity(network)
        nnmt.lif.exp.sensitivity_measure(network, frequency=frequency)
        nnmt.lif.exp.sensitivity_measure_all_eigenmodes(network, margin=margin)
        nnmt.lif.exp.power_spectra(network)
        # nnmt.lif.exp.additional_rates_for_fixed_input(
        #     network, mean_input_set, std_input_set)
        network.save(file=fixture_path + 'std_results.h5')

    if args['--firing_rates_fully_vectorized']:

        name = 'lif_exp/firing_rates_fully_vectorized'
        network = nnmt.models.Microcircuit(f'{config_path + name}.yaml')
        firing_rates = nnmt.lif.exp.firing_rates(network)
        network.save(f'{fixture_path + name}.h5')