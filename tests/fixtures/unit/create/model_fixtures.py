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
python tests/fixtures/create/model_fixtures.py -f

Usage: lif_fixtures.py [options]

Options:
    -f, --force        force code to run
    -h, --help         show this information
'''

import docopt
import sys
import nnmt.input_output as io
import nnmt

if __name__ == '__main__':
    # always show help message if not invoked with -f option
    if len(sys.argv) == 1:
        sys.argv.append('-h')
        
    args = docopt.docopt(__doc__)

    # only run code if users are sure they want to do it
    if '--force' in args.keys():

        fixture_path = 'unit/data/'
        config_path = 'unit/config/models/'
        
        network = nnmt.models.Microcircuit(
            config_path + 'network_params.yaml',
            config_path + 'analysis_params.yaml')
        
        nnmt.lif.exp.firing_rates(network)
        network.save(fixture_path + 'test_network.h5')
