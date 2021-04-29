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

        network = lmt.Network(config_path + 'network_params.yaml',
                              config_path + 'analysis_params.yaml')
        
        omega = network.analysis_params['omega']
        mean_input_set = network.network_params['mean_input_set']
        std_input_set = network.network_params['std_input_set']
        network.results['firing_rates_taylor'] = network.firing_rates(
            method='taylor')
        network.working_point(method='shift')
        network.delay_dist_matrix()
        network.delay_dist_matrix(omega)
        network.results['tf_taylor'] = network.transfer_function(
            method='taylor')
        network.results['tf_shift'] = network.transfer_function(
            method='shift')
        network.transfer_function(omega)
        network.sensitivity_measure(omega)
        network.power_spectra()
        network.results['eigenvalue_spectra_MH'] = (
            network.eigenvalue_spectra('MH'))
        network.results['r_eigenvec_spectra_MH'] = (
            network.r_eigenvec_spectra('MH'))
        network.results['l_eigenvec_spectra_MH'] = (
            network.l_eigenvec_spectra('MH'))
        network.results['eigenvalue_spectra_prop'] = (
            network.eigenvalue_spectra('prop'))
        network.results['r_eigenvec_spectra_prop'] = (
            network.r_eigenvec_spectra('prop'))
        network.results['l_eigenvec_spectra_prop'] = (
            network.l_eigenvec_spectra('prop'))
        network.results['eigenvalue_spectra_prop_inv'] = (
            network.eigenvalue_spectra('prop_inv'))
        network.results['r_eigenvec_spectra_prop_inv'] = (
            network.r_eigenvec_spectra('prop_inv'))
        network.results['l_eigenvec_spectra_prop_inv'] = (
            network.l_eigenvec_spectra('prop_inv'))
        network.additional_rates_for_fixed_input(mean_input_set,
                                                 std_input_set)
        network.save(file=fixture_path + 'std_results.h5')
