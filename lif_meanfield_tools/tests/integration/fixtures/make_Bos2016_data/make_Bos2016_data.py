"""make_Bos2016_data.py

Outputs:
    <filename>: full path including filename and extension of .txt file to return
Usage:
  calculate_PD_circuit_Bos_code.py output <filename>
  calculate_PD_circuit_Bos_code.py (-h | --help)
  calculate_PD_circuit_Bos_code.py --version
Options:
  -h --help     Show this screen.
  --version     Show version.
"""

import plot_helpers
import meanfield.circuit as circuit
import numpy as np
from docopt import docopt
import sys
import h5py_wrapper.wrapper as h5

if __name__ == '__main__':
    arguments = docopt(__doc__, version='0.1')
    filename = arguments['<filename>']

    # Original parameters extracted from the data + modification
    circ_params_with_modifications = plot_helpers.get_parameter_microcircuit()

    circ = circuit.Circuit('microcircuit',
                           circ_params_with_modifications,
                           analysis_type='dynamical',
                           fmax=500.0,
                           from_file=False)

    power_spectra_freqs, power_spectra = circ.create_power_spectra()

    exemplary_frequency_idx = 20
    omega = circ.omegas[exemplary_frequency_idx]
    print(omega)
    H = circ.ana.create_H(omega)
    MH = circ.ana.create_MH(omega)
    delay_dist = circ.ana.create_delay_dist_matrix(omega)

    transfer_function = circ.trans_func
    tau_s = circ.params['tauf'] * 1e-3
    transfer_function_with_synaptic_filter = np.copy(transfer_function)

    for i, trans_func in enumerate(transfer_function):
        for j, value in enumerate(trans_func):
            transfer_function_with_synaptic_filter[i,j] = value / complex(1., omegas[j] * tau_s)

    eigenvalue_spectra_freqs, eigenvalue_spectra = circ.create_eigenvalue_spectra('MH')

    h5.add_to_h5(filename, {'params': circ.params,
                            'omegas': circ.omegas,
                            'firing_rates': circ.th_rates,
                            'transfer_function': transfer_function,
                            'transfer_function_with_synaptic_filter': transfer_function_with_synaptic_filter,
                            'power_spectra': power_spectra,
                            'H': H,
                            'MH': MH,
                            'delay_dist': delay_dist,
                            'eigenvalue_spectra': eigenvalue_spectra,
                            'exemplary_frequency_idx': exemplary_frequency_idx
                            }
                 )


    # frequencies = [64, 241, 263, 267, 284]
    #
    # for f in frequencies:
    #
    #     Z = circ.get_sensitivity_measure(f)
    #
    # eigc = eigs[eig_index][np.argmin(abs(eigs[eig_index]-1))]
    #
    # k = np.asarray([1, 0])-np.asarray([eigc.real, eigc.imag])
    # k /= np.sqrt(np.dot(k, k))
    # k_per = np.asarray([-k[1], k[0]])
    # k_per /= np.sqrt(np.dot(k_per, k_per))
    # print(k, k_per)
