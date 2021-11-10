"""
Intermediate usage example
==========================

Here we show how to use NNMT to calculate the power spectra of the
:cite:t:`potjans2014` microcircuit model.
"""

import numpy as np
import matplotlib.pyplot as plt
import nnmt


###############################################################################
# First we try loading previous results, which makes running this script for
# the second time much faster. If there are no stored results, we instantiate
# a microcircuit model.
try:
    network = nnmt.models.Network(file='microcircuit.h5')
except IOError:
    network = nnmt.models.Microcircuit('network_params.yaml',
                                       'analysis_params.yaml')

###############################################################################
# We compute the working point, which only requires the network parameters
# defined in ``network_params.yaml`` (see minimal example as well).
print('Compute working point')
nnmt.lif.exp.working_point(network)

###############################################################################
# Then we continue by computing the transfer function, the delay distribution
# matrix, the effective connectivity, and finally the power spectrum. This
# requires the definition of the analysis frequencies in
# ``analysis_params.yaml``.
print('Compute transfer function')
nnmt.lif.exp.transfer_function(network)
print('Compute delay_dist_matrix')
nnmt.network_properties.delay_dist_matrix(network)
print('Compute effective connectivity')
nnmt.lif.exp.effective_connectivity(network)
print('Compute power spectra')
nnmt.lif.exp.power_spectra(network)

###############################################################################
# We retrieve the analysis frequencies and power spectra and plot the results
freqs = network.analysis_params['omegas'] / 2 / np.pi
spectra = network.results['lif.exp.power_spectra'].T

for i in range(8):
    plt.plot(freqs, spectra[i], label=network.network_params['populations'][i])
plt.yscale('log')
plt.xlabel('frequency (1/s)')
plt.ylabel('power')
plt.legend()
plt.show()

# save results to h5 file
network.save('microcircuit.h5')
