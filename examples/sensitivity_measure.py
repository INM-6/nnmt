# Minimal working example for sensitivity measure

import nnmt
from nnmt.__init__ import ureg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# instantiate network
network = nnmt.Network(network_params='network_params_microcircuit.yaml',
                      analysis_params='analysis_params.yaml')

network.transfer_function()
network.save()

fig = plt.figure(figsize=(10,3))
gs = gridspec.GridSpec(1,4)

labels = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']

# sensitivity measure
freqs = network.analysis_params['omegas']/2./np.pi
power = network.power_spectra()

pop_idx, freq_idx =  np.unravel_index(np.argmax(power), np.shape(power))
frequency = freqs[freq_idx]
print('frequency: ', frequency)
print('population: ', labels[pop_idx])

plt.title('%s Hz' %frequency)
sm =network.sensitivity_measure(freq=frequency)

eigs = network.eigenvalue_spectra('MH')

eigc = eigs[pop_idx][np.argmin(abs(eigs[pop_idx]-1))]

Z = network.sensitivity_measure(frequency)
k = np.asarray([1,0])-np.asarray([eigc.real,eigc.imag])
k /= np.sqrt(np.dot(k,k))
k_per = np.asarray([-k[1],k[0]])
k_per /= np.sqrt(np.dot(k_per,k_per))
Z_amp = Z.real*k[0]+Z.imag*k[1]
Z_freq = Z.real*k_per[0]+Z.imag*k_per[1]

titles = ['Re(Z)','Im(Z)', 'Z_amp', 'Z_freq']
for i,d in enumerate([np.real(sm), np.imag(sm), Z_amp, Z_freq]):
    ax = plt.subplot(gs[i])
    im = ax.imshow(d, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('sources')
    ax.set_ylabel('targets')
    ax.set_title(titles[i])
    if i>0:
        ax.set_yticklabels([])
        ax.set_ylabel('')

plt.savefig('sensitivity_measure.png')
plt.show()
