#!/usr/bin/env python
# encoding:utf8
'''
Creates fixtures for nnmt tests of binary neurons.

WARNING: Only use this script, if your code is trustworthy! The script runs
         the nnmt code to produce the fixtures that are then stored in h5
         format. If you run this script and your code is not working correctly,
         a lot of tests will pass despite your code giving wrong results.

If you still want to run this script, go to /nnmt/tests/fixture/ and type:
python unit/create/binary_fixtures.py -f <module>

Usage: binary_fixtures.py [options] <module>

Options:
    -f, --force        force code to run -h, --help         show this
    information
'''

import docopt
import sys
import os
import numpy as np

import nnmt

from helpers import (create_and_save_fixtures,
                     load_params_and_regimes)


if __name__ == '__main__':
    # always show help message if not invoked with -f option
    if len(sys.argv) == 1:
        sys.argv.append('-h')

    args = docopt.docopt(__doc__)

    # only run code if users are sure they want to do it
    if '--force' in args.keys():

        fixture_path = 'unit/data/'
        config_path_prefix = 'unit/config/binary/'
        neuron_prefix = 'binary_'

        module = args['<module>']

        run_calc = False

        if (module == 'mean_activity') or (module == 'all'):
            config_path = config_path_prefix + 'working_point/'
            regime_params, regimes = load_params_and_regimes(config_path)
            create_and_save_fixtures(
                nnmt.binary._mean_activity_for_given_input,
                regime_params, regimes,
                fixture_path + neuron_prefix + 'mean_activity.h5')

            run_calc = True

        if (module == 'inputs') or (module == 'all'):
            config_path = config_path_prefix + 'working_point/'
            regime_params, regimes = load_params_and_regimes(config_path)
            create_and_save_fixtures(
                nnmt.binary._mean_input,
                regime_params, regimes,
                fixture_path + neuron_prefix + 'mean_input.h5')
            create_and_save_fixtures(
                nnmt.binary._std_input,
                regime_params, regimes,
                fixture_path + neuron_prefix + 'std_input.h5')

            run_calc = True

        if (module == 'balanced_threshold') or (module == 'all'):
            config_path = config_path_prefix + 'working_point/'
            regime_params, regimes = load_params_and_regimes(config_path)
            create_and_save_fixtures(
                nnmt.binary._mean_input,
                regime_params, regimes,
                fixture_path + neuron_prefix + 'balanced_threshold.h5')

            run_calc = True

        if not run_calc:
            print('No such module')
