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
import h5py_wrapper as h5

import lif_meanfield_tools.lif.delta.static as delta
from lif_meanfield_tools.input_output import load_val_unit_dict_from_yaml
from lif_meanfield_tools.utils import _strip_units


def fixtures_firing_rate(path, regimes, regime_params):
    results = {}
    for regime, params in zip(regimes, regime_params):
        results[regime] = {
            'params': params,
            'output': delta._firing_rate(**params)
            }
    file = path + 'lif_delta_firing_rate.h5'
    h5.save(file, results, overwrite_dataset=True)


if __name__ == '__main__':
    # always show help message if not invoked with -f option
    if len(sys.argv) == 1:
        sys.argv.append('-h')

    args = docopt.docopt(__doc__)

    # only run code if users are sure they want to do it
    if '--force' in args.keys():

        fixture_path = 'unit/data/'
        config_path = 'unit/config/'
        
        regimes = [
            'sub_threshold_weak_noise',
            'supra_threshold_weak_noise',
            'sub_threshold_strong_noise',
            'supra_threshold_strong_noise',
            ]
        param_files = [config_path + regime + '.yaml'
                         for regime in regimes]
        regime_params = [load_val_unit_dict_from_yaml(file)
                         for file in param_files]
        [_strip_units(dict) for dict in regime_params]
            
        fixtures_firing_rate(fixture_path, regimes, regime_params)
