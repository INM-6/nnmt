#!/usr/bin/env python
# encoding:utf8
'''
Creates fixtures for lif_meanfield_tools tests.

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
import numpy as np
import mpmath
import sys

from lif_meanfield_tools.aux_calcs import (
    Phi,
    Phi_prime_mu,
    Psi,
    d_Psi,
    d_2_Psi,
    p_hat_boxcar
    )


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
        pcfu_outputs.append(mpmath.pcfu(z, -x))

    outputs = []
    for z, x in zip(zs, xs):
        outputs.append(Psi(z, x))
    
    np.savez(output_file, zs=zs, xs=xs, pcfus=pcfu_outputs,
             outputs=outputs)
    
    
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

    
if __name__ == '__main__':
    # always show help message if not invoked with -f option
    if len(sys.argv) == 1:
        sys.argv.append('-h')
        
    args = docopt.docopt(__doc__)

    # only run code if users are sure they want to do it
    if '--force' in args.keys():
        fixture_path = 'tests/fixtures/data/'

        fix_Phi(fixture_path)
        fix_Phi_prime_mu(fixture_path)
        fix_Psi(fixture_path)
        fix_d_Psi(fixture_path)
        fix_d_2_Psi(fixture_path)
        fix_p_hat_boxcar(fixture_path)
