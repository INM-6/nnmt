#!/usr/bin/env python
# encoding:utf8
'''
Creates fixtures for nnmt tests.

WARNING: Only use this script, if your code is trustworthy! The script runs
         the nnmt code to produce the fixtures that are then
         stored in h5 format. If you run this script and your code is not
         working correctly, a lot of tests will pass despite your code giving
         wrong results.

If you still want to run this script type:
python tests/fixtures/create/lif_fixtures.py -f <module>

Usage: lif_fixtures.py [options] <module>

Options:
    -f, --force        force code to run
    -h, --help         show this information
'''

import docopt
import inspect
import sys
import os
import h5py_wrapper as h5

from nnmt.input_output import load_val_unit_dict_from_yaml
from nnmt.utils import (
    _strip_units,
    _to_si_units,
    )

import nnmt


def get_required_params(func, all_params):
    """Checks arguments of func and returns corresponding parameters."""
    required_keys = list(inspect.signature(func).parameters)
    required_params = {k: v for k, v in all_params.items()
                       if k in required_keys}
    return required_params


def extract_required_params(func, regime_params):
    extracted = [get_required_params(func, params)
                 for params in regime_params]
    return extracted
    

def create_and_save_fixtures(func, regime_params, regimes, file):
    results = {}
    regime_params = extract_required_params(func,
                                            regime_params)
    for regime, params in zip(regimes, regime_params):
        output = func(**params)
        results[regime] = {
            'params': params,
            'output': output
            }
    h5.save(file, results, overwrite_dataset=True)
    

if __name__ == '__main__':
    # always show help message if not invoked with -f option
    if len(sys.argv) == 1:
        sys.argv.append('-h')

    args = docopt.docopt(__doc__)

    # only run code if users are sure they want to do it
    if '--force' in args.keys():
        
        fixture_path = 'unit/data/'
        
        module = args['<module>']
        config_path = 'unit/config/' + module + '/'
        
        param_files = os.listdir(config_path)
        regimes = [file.replace('.yaml', '') for file in param_files]
        regime_params = [load_val_unit_dict_from_yaml(config_path + file)
                         for file in param_files]
        for dict in regime_params:
            _to_si_units(dict)
            _strip_units(dict)
        
        if module == 'firing_rates':
            create_and_save_fixtures(nnmt.lif.delta._firing_rate,
                                     regime_params, regimes,
                                     fixture_path + 'lif_delta_firing_rate.h5')
            create_and_save_fixtures(nnmt.lif.exp._firing_rate_taylor,
                                     regime_params, regimes,
                                     fixture_path
                                     + 'lif_exp_firing_rate_taylor.h5')
            create_and_save_fixtures(nnmt.lif.exp._firing_rate_shift,
                                     regime_params, regimes,
                                     fixture_path
                                     + 'lif_exp_firing_rate_shift.h5')
        elif module == 'inputs':
            create_and_save_fixtures(nnmt.lif._static._mean_input,
                                     regime_params, regimes,
                                     fixture_path + 'lif_mean_input.h5')
            create_and_save_fixtures(nnmt.lif._static._std_input,
                                     regime_params, regimes,
                                     fixture_path + 'lif_std_input.h5')
        elif module == 'transfer_functions':
            create_and_save_fixtures(nnmt.lif.exp._transfer_function_shift,
                                     regime_params, regimes,
                                     fixture_path
                                     + 'lif_exp_transfer_function_shift.h5')
            create_and_save_fixtures(nnmt.lif.exp._transfer_function_taylor,
                                     regime_params, regimes,
                                     fixture_path
                                     + 'lif_exp_transfer_function_taylor.h5')
        elif module == 'sensitivity_measure':
            create_and_save_fixtures(nnmt.lif.exp._effective_connectivity,
                                     regime_params, regimes,
                                     fixture_path
                                     + 'lif_exp_effective_connectivity.h5')
            create_and_save_fixtures(nnmt.lif.exp._sensitivity_measure,
                                     regime_params, regimes,
                                     fixture_path
                                     + 'lif_exp_sensitivity_measure.h5')
            create_and_save_fixtures(nnmt.lif.exp._power_spectra,
                                     regime_params, regimes,
                                     fixture_path
                                     + 'lif_exp_power_spectra.h5')
            create_and_save_fixtures(nnmt.lif.exp._propagator,
                                     regime_params, regimes,
                                     fixture_path
                                     + 'lif_exp_propagator.h5')
        elif module == 'external_rates':
            create_and_save_fixtures(
                nnmt.lif.exp._external_rates_for_fixed_input,
                regime_params, regimes,
                fixture_path
                + 'lif_exp_external_rates_for_fixed_input.h5')
        else:
            print('No such module')
