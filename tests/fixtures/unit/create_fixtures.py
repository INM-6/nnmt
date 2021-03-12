#!/usr/bin/env python
# encoding:utf8
'''
Creates fixtures for lif_meanfield_tools tests.

WARNING: Only use this script, if your code is trustworthy! The script runs
         the lif_meanfield_tools code to produce the fixtures that are then
         stored in h5 format. If you run this script and your code is not
         working correctly, a lot of tests will pass despite your code giving
         wrong results.

If you still want to run this script type: python tests/fixtures/create_fixtures.py -f

Usage: create_fixtures.py [options]

Options:
    -f, --force        force code to run
    -h, --help         show this information
'''

import docopt
import numpy as np
import mpmath
import sys

import lif_meanfield_tools as lmt
from lif_meanfield_tools.meanfield_calcs import (
    effective_coupling_strength,
    )
from lif_meanfield_tools.aux_calcs import (
    d_nu_d_mu,
    d_nu_d_mu_fb433,
    d_nu_d_nu_in_fb,
    Phi,
    Phi_prime_mu,
    Psi,
    d_Psi,
    d_2_Psi,
    p_hat_boxcar
    )

ureg = lmt.ureg


def fix_additional_rates_for_fixed_input(network, file):
    """Call additional_rates_for_fixed_input and save results as h5."""
    nu_e_ext, nu_i_ext = network.additional_rates_for_fixed_input(
        network.network_params['mu_set'],
        network.network_params['sigma_set'])
    network.results['add_nu_e_ext'] = nu_e_ext
    network.results['add_nu_i_ext'] = nu_i_ext
    network.save(file_name=file)


def fix_d_nu_d_mu(network, file):
    """Call d_nu_d_mu and save results using network.save() as h5."""
    params = network.network_params
    mus = network.mean_input()
    sigmas = network.std_input()
    results = [d_nu_d_mu(params['tau_m'].magnitude,
                         params['tau_r'].magnitude,
                         params['V_th_rel'].magnitude,
                         params['V_0_rel'].magnitude,
                         mu, sigma)
               for mu, sigma in zip(mus.magnitude, sigmas.magnitude)]
    network.results['d_nu_d_mu'] = results
    network.save(file_name=file)


def fix_d_nu_d_mu_fb433(network, file):
    """Call d_nu_d_mu_fb433 and save results using network.save() as h5."""
    params = network.network_params
    mus = network.mean_input()
    sigmas = network.std_input()
    results = [d_nu_d_mu_fb433(params['tau_m'],
                               params['tau_s'],
                               params['tau_r'],
                               params['V_th_rel'],
                               params['V_0_rel'],
                               mu, sigma)
               for mu, sigma in zip(mus, sigmas)]
    network.results['d_nu_d_mu_fb433'] = results
    network.save(file_name=file)


def fix_d_nu_d_nu_in_fb(network, file):
    """Call d_nu_d_mu_in_fb and save results using network.save() as h5."""
    params = network.network_params
    mus = network.mean_input()
    sigmas = network.std_input()
    results = [d_nu_d_nu_in_fb(params['tau_m'],
                               params['tau_s'],
                               params['tau_r'],
                               params['V_th_rel'],
                               params['V_0_rel'],
                               params['j'],
                               mu, sigma)
               for mu, sigma in zip(mus, sigmas)]
    network.results['d_nu_d_nu_in_fb'] = results
    network.save(file_name=file)


def fix_d_Psi(fixture_path):
    """Call d_Psi for a range of possible inputs and save result as fixture."""
    function_name = 'd_Psi'
    output_file = fixture_path + function_name + '.npz'

    z_range = np.concatenate([-np.logspace(2, -5, 4), [0],
                              np.logspace(-5, 2, 4)])
    a, b = np.meshgrid(z_range, z_range)
    zs = a.flatten() + complex(0, 1) * b.flatten()
    xs = np.linspace(-10, 10, 8)

    zs, xs = np.meshgrid(zs, xs)
    zs = zs.flatten()
    xs = xs.flatten()

    psi_outputs = []
    for z, x in zip(zs, xs):
        psi_outputs.append(Psi(z + 1, x))

    outputs = []
    for z, x in zip(zs, xs):
        outputs.append(d_Psi(z, x))

    np.savez(output_file, zs=zs, xs=xs, psis=psi_outputs, outputs=outputs)


def fix_d_2_Psi(fixture_path):
    """Call d_2_Psi for a range of inputs and save result as fixture."""
    function_name = 'd_2_Psi'
    output_file = fixture_path + function_name + '.npz'

    z_range = np.concatenate([-np.logspace(2, -5, 4), [0],
                              np.logspace(-5, 2, 4)])
    a, b = np.meshgrid(z_range, z_range)
    zs = a.flatten() + complex(0, 1) * b.flatten()
    xs = np.linspace(-10, 10, 8)

    zs, xs = np.meshgrid(zs, xs)
    zs = zs.flatten()
    xs = xs.flatten()

    psi_outputs = []
    for z, x in zip(zs, xs):
        psi_outputs.append(Psi(z + 2, x))

    outputs = []
    for z, x in zip(zs, xs):
        outputs.append(d_2_Psi(z, x))

    np.savez(output_file, zs=zs, xs=xs, psis=psi_outputs, outputs=outputs)


def fix_delay_dist_single(network, file):
    """Calculate delay_dist_matrix for a single freq and save as h5."""
    network.delay_dist_matrix(network.analysis_params['omega'])
    network.save(file_name=file)


def fix_delay_dist_matrix(network, file):
    """Calculate fixtures for all delay dist matrix options and save as h5."""
    original_delay_dist = network.network_params['delay_dist']
    network = network.change_parameters(
        changed_network_params={'delay_dist': 'none'})
    dd_none = network.delay_dist_matrix()
    network = network.change_parameters(
        changed_network_params={'delay_dist': 'truncated_gaussian'})
    dd_truncated_gaussian = network.delay_dist_matrix()
    network = network.change_parameters(
        changed_network_params={'delay_dist': 'gaussian'})
    dd_gaussian = network.delay_dist_matrix()
    network.results['delay_dist_none'] = dd_none
    network.results['delay_dist_truncated_gaussian'] = dd_truncated_gaussian
    network.results['delay_dist_gaussian'] = dd_gaussian
    network.network_params['delay_dist'] = original_delay_dist
    network.save(file_name=file)


def fix_eff_coupling_strength(network, file):
    """Calc eff_coupling_strength and save as h5."""
    eff_coupling_strength = effective_coupling_strength(
        network.network_params['tau_m'],
        network.network_params['tau_s'],
        network.network_params['tau_r'],
        network.network_params['V_0_rel'],
        network.network_params['V_th_rel'],
        network.network_params['J'],
        network.results['mean_input'],
        network.results['std_input'])
    network.results['effective_coupling_strength'] = eff_coupling_strength
    network.save(file_name=file)


def fix_eigenspectra(network, file):
    """Calc eigenvalues, l and r eigenvecs and save as h5."""
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

    network.save(file_name=file)


def fix_p_hat_boxcar(fixture_path):
    """Call p_hat_boxcar for a range of inputs and save result as fixture."""
    function_name = 'p_hat_boxcar'
    output_file = fixture_path + function_name + '.npz'

    lp = -5
    hp = 5
    steps = 20
    ks = np.concatenate([-np.logspace(hp, lp, steps), [0],
                         np.logspace(lp, hp, steps)])

    widths = np.logspace(-5, 5)

    ks, widths = np.meshgrid(ks, widths)
    ks = ks.flatten()
    widths = widths.flatten()

    outputs = []

    for k, width in zip(ks, widths):
        outputs.append(p_hat_boxcar(k, width))

    np.savez(output_file, ks=ks, widths=widths, outputs=outputs)


def fix_Phi(fixture_path):
    """Call Phi for a range of possible inputs and save result as fixture."""
    function_name = 'Phi'
    output_file = fixture_path + function_name + '.npz'

    lp = -5
    hp = 1.5
    test_inputs = np.concatenate([-np.logspace(hp, lp),
                                  [0],
                                  np.logspace(lp, hp)])

    outputs = []
    for test_input in test_inputs:
        outputs.append(Phi(test_input))

    np.savez(output_file, s_values=test_inputs, outputs=outputs)


def fix_Phi_prime_mu(fixture_path):
    """Call Phi_prime_mu for a range of inputs and save result as fixture."""
    function_name = 'Phi_prime_mu'
    output_file = fixture_path + function_name + '.npz'

    lp = -5
    hp = 1.5
    steps = 20
    s_values = np.concatenate([-np.logspace(hp, lp, steps),
                               [0],
                               np.logspace(lp, hp, steps)])
    sigmas = np.linspace(1, 100, 10)

    s_values, sigmas = np.meshgrid(s_values, sigmas)
    s_values = s_values.flatten()
    sigmas = sigmas.flatten()

    outputs = []
    for s, sigma in zip(s_values, sigmas):
        outputs.append(Phi_prime_mu(s, sigma))

    np.savez(output_file, s_values=s_values, sigmas=sigmas,
             outputs=outputs)


def fix_power_spectra(network, file):
    network.power_spectra()
    network.save(file_name=file)


def fix_Psi(fixture_path):
    """Call Psi for a range of possible inputs and save result as fixture."""
    function_name = 'Psi'
    output_file = fixture_path + function_name + '.npz'

    z_range = np.concatenate([-np.logspace(2, -5, 4), [0],
                              np.logspace(-5, 2, 4)])
    a, b = np.meshgrid(z_range, z_range)
    zs = a.flatten() + complex(0, 1) * b.flatten()
    xs = np.linspace(-10, 10, 8)

    zs, xs = np.meshgrid(zs, xs)
    zs = zs.flatten()
    xs = xs.flatten()

    pcfu_outputs = []
    for z, x in zip(zs, xs):
        result = mpmath.pcfu(z, -x)
        pcfu_outputs.append(complex(result.real, result.imag))

    outputs = []
    for z, x in zip(zs, xs):
        outputs.append(Psi(z, x))
    np.savez(output_file, zs=zs, xs=xs, pcfus=pcfu_outputs,
             outputs=outputs)


def fix_sensitivity_measure(network, file):
    """Calc sensitivity_measure and save as h5 using network.save()."""
    omega = network.analysis_params['omega']
    network.sensitivity_measure(omega)
    network.transfer_function(omega)
    network.save(file_name=file)


def fix_transfer_function(network, file):
    """Calculate results for all options of transfer_function."""
    network.transfer_function(method='shift')
    network.transfer_function(method='taylor')
    tfs = network.results['transfer_function']
    network.results['tf_shift'] = tfs[0]
    network.results['tf_taylor'] = tfs[1]
    network.results['transfer_function'] = tfs[0]
    network.save(file_name=file)


def fix_working_point(network, file):
    """Calculate working_point and save results as h5 using network.save()."""
    network.working_point()
    network.save(file_name=file)


if __name__ == '__main__':
    # always show help message if not invoked with -f option
    if len(sys.argv) == 1:
        sys.argv.append('-h')

    args = docopt.docopt(__doc__)

    # only run code if users are sure they want to do it
    if '--force' in args.keys():

        fixture_path = 'tests/fixtures/unit/data/'
        config_path = 'tests/fixtures/unit/config/'

        # purely numerical fixtures, that don't need any network
        # fix_Phi(fixture_path)
        # fix_Phi_prime_mu(fixture_path)
        # fix_Psi(fixture_path)
        # fix_d_Psi(fixture_path)
        # fix_d_2_Psi(fixture_path)
        # fix_p_hat_boxcar(fixture_path)

        configs = dict(
            noise_driven=(config_path + 'network_params_microcircuit.yaml'),
            mean_driven=(config_path + 'mean_driven.yaml'),
            )
        analysis_param_file = config_path + 'analysis_params_test.yaml'
        for regime, param_file in configs.items():

            file_path = '{}{}_regime.h5'.format(fixture_path, regime)

            network = lmt.Network(param_file, analysis_param_file)
            network.network_params['regime'] = regime

            # fixtures that need a network, or network params to be calculated
            fix_working_point(network, file_path)
            fix_transfer_function(network, file_path)
            fix_delay_dist_single(network, file_path)
            fix_delay_dist_matrix(network, file_path)
            fix_sensitivity_measure(network, file_path)
            fix_power_spectra(network, file_path)
            fix_eigenspectra(network, file_path)
            fix_additional_rates_for_fixed_input(network, file_path)
            fix_eff_coupling_strength(network, file_path)
            fix_d_nu_d_mu(network, file_path)
            fix_d_nu_d_mu_fb433(network, file_path)
            fix_d_nu_d_nu_in_fb(network, file_path)
