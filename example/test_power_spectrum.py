# Minimal working example to test power spectra

import lif_meanfield_tools as lmt
from lif_meanfield_tools.__init__ import ureg
import numpy as np
import matplotlib.pyplot as plt

# instantiate network
network = lmt.Network(network_params='network_params_microcircuit.yaml',
                      analysis_params='analysis_params.yaml')

network.transfer_function()
network.save()

freqs = network.analysis_params['omegas']/2./np.pi
power = network.power_spectra()
network.save()

plt.figure()
for i in range(8):
    plt.plot(freqs, power[i], label=network.network_params['populations'][i])
plt.yscale('log')
plt.legend()

plt.show()
