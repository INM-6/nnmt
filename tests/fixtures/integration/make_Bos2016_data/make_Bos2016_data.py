"""
This script reproduces the data produces and analyzed in the following 
publication:

Bos, H., Diesmann, M. & Helias, M. 
Identifying Anatomical Origins of Coexisting Oscillations in the Cortical 
Microcircuit. 
PLOS Computational Biology 12, 1-34 (2016).
"""

import plot_helpers
import meanfield.circuit as circuit
import numpy as np
import h5py_wrapper.wrapper as h5


fix_path = 'integration/data/'


if __name__ == '__main__':
    filename = fix_path + 'Bos2016_data.h5'

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
            transfer_function_with_synaptic_filter[i, j] = (
                value / complex(1., circ.omegas[j] * tau_s))

    eigenvalue_spectra_freqs, eigenvalue_spectra = (
        circ.create_eigenvalue_spectra('MH'))

    h5.save(filename, {'params': circ.params,
                            'omegas': circ.omegas,
                            'firing_rates': circ.th_rates,
                            'transfer_function': transfer_function,
                            'transfer_function_with_synaptic_filter':
                                transfer_function_with_synaptic_filter,
                            'power_spectra': power_spectra,
                            'H': H,
                            'MH': MH,
                            'delay_dist': delay_dist,
                            'eigenvalue_spectra': eigenvalue_spectra,
                            'exemplary_frequency_idx': exemplary_frequency_idx
                            }, overwrite_dataset=True
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
