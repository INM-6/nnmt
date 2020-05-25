#!/usr/bin/env python
# encoding:utf8
'''
Creates fixtures for lif_meanfield_tools tests.

WARNING: Only use this script, if your code is trustworthy! The script runs
         the code to produce the fixtures that are then stored in h5 format.
         If you run this script and your code is not working correctly, most
         tests will pass despite your code giving wrong results.

If you still want to run this script type: python create_fixtures.py -f

Usage: create_fixtures.py [options]

Options:
    -f, --force        force code to run
    -h, --help         show this information
'''

import docopt
import sys

import lif_meanfield_tools as lmt
ureg = lmt.ureg


def fixture_working_point(network):
    network.working_point()


def fixture_transfer_function(network):
    """Calculate results for all options of transfer_function."""
    network.transfer_function(method='shift')
    network.results['tf_shift'] = network.results.pop('transfer_function')
    network.transfer_function(method='taylor')
    network.results['tf_taylor'] = network.results['transfer_function']


def fixture_delay_dist_matrix(network):
    """Calculate fixtures for all delay dist matrix options."""
    original_delay_dist = network.network_params['delay_dist']
    network.network_params['delay_dist'] = 'none'
    network.delay_dist_matrix()
    dd_none = network.results['delay_dist']
    network.network_params['delay_dist'] = 'truncated_gaussian'
    network.delay_dist_matrix()
    dd_truncated_gaussian = network.results.pop('delay_dist')
    network.network_params['delay_dist'] = 'gaussian'
    network.delay_dist_matrix()
    dd_gaussian = network.results.pop('delay_dist')
    network.results['delay_dist_none'] = dd_none
    network.results['delay_dist_truncated_gaussian'] = dd_truncated_gaussian
    network.results['delay_dist_gaussian'] = dd_gaussian
    network.network_params['delay_dist'] = original_delay_dist
    
    
def fixture_eigenspectra(network):
    regime = network.network_params['regime']
    
    network.eigenvalue_spectra('MH')
    network.eigenvalue_spectra('prop')
    # inverse propagator does not exist in neg rate regime with current params!
    if regime != 'negative_firing_rate':
        network.eigenvalue_spectra('prop_inv')

    network.r_eigenvec_spectra('MH')
    network.r_eigenvec_spectra('prop')
    # inverse propagator does not exist in neg rate regime with current params!
    if regime != 'negative_firing_rate':
        network.r_eigenvec_spectra('prop_inv')

    network.l_eigenvec_spectra('MH')
    network.l_eigenvec_spectra('prop')
    # inverse propagator does not exist in neg rate regime with current params!
    if regime != 'negative_firing_rate':
        network.l_eigenvec_spectra('prop_inv')
    
    
def fixture_power_spectra(network):
    network.power_spectra()
    
    
def fixture_sensitivity_measure(network):
    omega = network.analysis_params['omega']
    network.sensitivity_measure(omega)
    network.transfer_function(omega)
    

def fixture_additional_rates_for_fixed_input(network):
    nu_e_ext, nu_i_ext = network.additional_rates_for_fixed_input(
        network.network_params['mean_input_set'],
        network.network_params['std_input_set'])
    network.results['add_nu_e_ext'] = nu_e_ext
    network.results['add_nu_i_ext'] = nu_i_ext
    

def fixture_eff_coupling_strength(network):
    eff_coupling_strength = lmt.meanfield_calcs.effective_coupling_strength(
        network.network_params['tau_m'],
        network.network_params['tau_s'],
        network.network_params['tau_r'],
        network.network_params['V_0_rel'],
        network.network_params['V_th_rel'],
        network.network_params['J'],
        network.results['mean_input'],
        network.results['std_input'])
    network.results['effective_coupling_strength'] = eff_coupling_strength


def fixture_d_nu_d_mu(network):
    params = network.network_params
    mus = network.mean_input()
    sigmas = network.std_input()
    d_nu_d_mu = [lmt.aux_calcs.d_nu_d_mu(params['tau_m'],
                                         params['tau_r'],
                                         params['V_th_rel'],
                                         params['V_0_rel'],
                                         mu, sigma)
                 for mu, sigma in zip(mus, sigmas)]
    network.results['d_nu_d_mu'] = d_nu_d_mu


def fixture_d_nu_d_mu_fb433(network):
    params = network.network_params
    mus = network.mean_input()
    sigmas = network.std_input()
    d_nu_d_mu_fb433 = [lmt.aux_calcs.d_nu_d_mu_fb433(params['tau_m'],
                                                     params['tau_s'],
                                                     params['tau_r'],
                                                     params['V_th_rel'],
                                                     params['V_0_rel'],
                                                     mu, sigma)
                       for mu, sigma in zip(mus, sigmas)]
    network.results['d_nu_d_mu_fb433'] = d_nu_d_mu_fb433


def fixture_d_nu_d_nu_in_fb(network):
    params = network.network_params
    mus = network.mean_input()
    sigmas = network.std_input()
    d_nu_d_nu_in_fb = [lmt.aux_calcs.d_nu_d_nu_in_fb(params['tau_m'],
                                                     params['tau_s'],
                                                     params['tau_r'],
                                                     params['V_th_rel'],
                                                     params['V_0_rel'],
                                                     params['j'],
                                                     mu, sigma)
                       for mu, sigma in zip(mus, sigmas)]
    network.results['d_nu_d_nu_in_fb'] = d_nu_d_nu_in_fb


configs = dict(noise_driven='network_params_microcircuit.yaml',
               negative_firing_rate='minimal_negative.yaml',)
               # mean_driven='small_network.yaml')

analysis_param_file = 'analysis_params_test.yaml'

if __name__ == '__main__':
    # always show help message if not invoked with -f option
    if len(sys.argv) == 1:
        sys.argv.append('-h')
        
    args = docopt.docopt(__doc__)

    # only run code if users are sure they want to do it
    if '--force' in args.keys():

        for regime, param_file in configs.items():
            
            network = lmt.Network(param_file, analysis_param_file)
        
            network.network_params['regime'] = regime
            #
            fixture_working_point(network)
            fixture_transfer_function(network)
            fixture_delay_dist_matrix(network)
            fixture_sensitivity_measure(network)
            fixture_power_spectra(network)
            fixture_additional_rates_for_fixed_input(network)
            fixture_eff_coupling_strength(network)
            fixture_d_nu_d_mu(network)
            fixture_d_nu_d_mu_fb433(network)
            fixture_d_nu_d_nu_in_fb(network)
        
            network.save(file_name='data/{}_regime.h5'.format(regime))
