"""compare_parameters.py

Outputs:
    <filename>: full path including filename and extension of .txt file to return

Usage:
  compare_parameters.py output <filename>
  compare_parameters.py (-h | --help)
  compare_parameters.py --version
Options:
  -h --help     Show this screen.
  --version     Show version.
"""

import plot_helpers
import meanfield.circuit as circuit
import numpy as np
from docopt import docopt
import sys

if __name__ == '__main__':
    arguments = docopt(__doc__, version='0.1')
    filename = arguments['<filename>']

    # New parameters used for Fig. 1
    circ_params_new = plot_helpers.get_parameter_microcircuit()

    # Original parameters extracted from the data
    circ = circuit.Circuit('microcircuit', analysis_type=None)
    circ_params_old = circ.params

    file = open(filename, 'w')
    sys.stdout = file

    for attribute in circ_params_new.keys():
        print(attribute)
        print('old/original:')
        print(circ_params_old[attribute])
        print('new:')
        print(circ_params_new[attribute])
        print('\n')

    attribute = 'I'
    print('old/original:')
    print(circ_params_old[attribute])
    print('new:')
    print(circ_params_new[attribute])
    print('difference:')
    print(circ_params_new[attribute] - circ_params_old[attribute])
    print('\n')

    print('Indegrees 4I to 4I:')
    print('old/original:')
    print(circ_params_old[attribute][2][3])
    print('new:')
    print(circ_params_new[attribute][2][3])
    print('diff = 15 %:')
    print(circ_params_old[attribute][2][3]/100*15)
    print('\n')

    file.close()
