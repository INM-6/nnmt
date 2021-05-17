#!/usr/bin/env python
# encoding:utf8
'''
Creates fixtures for lif_meanfield_tools integration tests.

WARNING: Only use this script, if your code is trustworthy! The script runs
         the lif_meanfield_tools code to produce the fixtures that are then
         stored in h5 format. If you run this script and your code is not
         working correctly, a lot of tests will pass despite your code giving
         wrong results.

If you still want to run this script type: python create_fixtures.py -f

Usage: create_fixtures.py [options]

Options:
    -f, --force        force code to run
    -h, --help         show this information
'''

import docopt
import sys

import lif_meanfield_tools as lmt


if __name__ == '__main__':
    # always show help message if not invoked with -f option
    if len(sys.argv) == 1:
        sys.argv.append('-h')
        
    args = docopt.docopt(__doc__)

    # only run code if users are sure they want to do it
    if '--force' in args.keys():

        fixture_path = 'integration/data/'
        config_path = 'integration/config/'

        network = lmt.networks.Microcircuit(
            config_path + 'network_params.yaml',
            config_path + 'analysis_params.yaml')
        
        omega = network.analysis_params['omega']
        mean_input_set = network.network_params['mean_input_set']
        std_input_set = network.network_params['std_input_set']
        network.results[
            'lif.exp.firing_rates_taylor'] = lmt.lif.exp.firing_rates(
            network, method='taylor')
        lmt.lif.exp.working_point(network, method='shift')
        network.results[
            'delay_dist_matrix'] = lmt.networks.utils.delay_dist_matrix(
                network)
        network.results['lif.exp.tf_single'] = lmt.lif.exp.transfer_function(
            network, omega)
        network.results['lif.exp.tf_taylor'] = lmt.lif.exp.transfer_function(
            network, method='taylor')
        network.results['lif.exp.tf_shift'] = lmt.lif.exp.transfer_function(
            network, method='shift')
        lmt.lif.exp.effective_connectivity(network)
        lmt.lif.exp.sensitivity_measure(network)
        lmt.lif.exp.power_spectra(network)
        # lmt.lif.exp.additional_rates_for_fixed_input(
        #     network, mean_input_set, std_input_set)
        network.save(file=fixture_path + 'std_results.h5', overwrite=True)
