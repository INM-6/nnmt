"""
Microcircuit Firing Rates
=========================

Here we calculate the firing rates of the :cite:t:`potjans2014` microcircuit
model.
"""

import nnmt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# use matplotlib style file
plt.style.use('frontiers.mplstyle')

###############################################################################
# First we create a network model of the microcircuit, passing the parameter
# yaml file.
microcircuit = nnmt.models.Microcircuit('network_params_microcircuit.yaml')

###############################################################################
# Then we simply calculate the firing rates for exponentially shape post
# synaptic currents, by calling the respective function and passing the
# microcircuit. Here we chose to use the 'taylor' method for calculating the
# firing rates.
firing_rates = nnmt.lif.exp.firing_rates(microcircuit, method='shift')

print(f'Mean rates: {firing_rates}')

###############################################################################
# Then we compare the rates to the publicated data from :cite:t:`bos2016`.
# simulated_rates = np.array([0.74460773, 2.69596288, 4.11150391, 5.62804937,
#                              6.63713466, 8.29040221, 1.1003033 , 7.66250752])
simulated_rates = np.array([0.943, 3.026, 4.368, 5.882,
                            7.733, 8.664, 1.096, 7.851])
print(f'Mean simulated rates: {simulated_rates}')

###############################################################################
# Finally, we plot the rates together in one plot.
fig = plt.figure(figsize=(3.34646, 3.34646 / 2),
                #  tight_layout=True)
                 constrained_layout=True)
#
# ax0 = fig.add_subplot(211)
ax = fig.add_subplot(111)
bars = ax.bar(np.arange(8), simulated_rates, align='center', color='grey',
              edgecolor='black', label='sim')

for i in [0, 2, 4, 6]:
    bars[i].set_color('black')

nnmt_handle = ax.scatter(np.arange(8), firing_rates, marker='X', color='r', s=50,
                         zorder=10, label='NNMT')
ax.set_xticks(np.arange(8))
ax.set_xticklabels(['2/3E', '2/3I', '4E', '4I', '5E', '5I', '6E', '6I'])
ax.set_yticks([1, 3, 5, 7])
ax.set_ylabel(r'$\bar{\nu}\,(1/s)$')

# exc_handle = mpatches.Patch(color='black', label='sim exc')
# inh_handle = mpatches.Patch(color='grey', label='sim inh')
plt.legend([bars[0], nnmt_handle, bars[1]], [None, 'NNMT', 'sim'],
           loc='upper left', fontsize=10, ncol=2, columnspacing=-2.8, handletextpad=0.2)

plt.savefig('microcircuit_rates.png')
