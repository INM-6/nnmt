"""
Microcircuit Firing Rates
=========================

Here we calculate the firing rates of the :cite:t:`potjans2014` microcircuit
model.
"""

import nnmt
import numpy as np
import matplotlib.pyplot as plt

# use matplotlib style file
plt.style.use('frontiers.mplstyle')

###############################################################################
# First we create a network model of the microcircuit, passing the parameter
# yaml file.
microcircuit = nnmt.models.Microcircuit(
    'parameters/network_params_microcircuit.yaml')

###############################################################################
# Then we simply calculate the firing rates for exponentially shape post
# synaptic currents, by calling the respective function and passing the
# microcircuit. Here we chose to use the 'taylor' method for calculating the
# firing rates.
firing_rates = nnmt.lif.exp.firing_rates(microcircuit, method='taylor')

print(f'Mean rates: {firing_rates}')

###############################################################################
# Then we compare the rates to the publicated data from :cite:t:`bos2016`.
simulated_rates = np.array([0.74460773, 2.69596288, 4.11150391, 5.62804937,
                            6.63713466, 8.29040221, 1.1003033, 7.66250752])
print(f'Mean simulated rates: {simulated_rates}')

###############################################################################
# Finally, we plot the rates together in one plot.
fig = plt.figure(figsize=(3.34646, 3.34646/2),
                 constrained_layout=True)

ax = fig.add_subplot(111)
bars = ax.bar(np.arange(8), simulated_rates, align='center', color='grey',
              edgecolor='black')

for i in [0, 2, 4, 6]:
    bars[i].set_color('black')

ax.scatter(np.arange(8), firing_rates, marker='X', color='r', s=50,
           zorder=10)
ax.set_xticks([0.5, 2.5, 4.5, 6.5])
ax.set_xticklabels(['L2/3', 'L4', 'L5', 'L6'])
ax.set_yticks([1, 3, 5, 7])
ax.set_ylabel(r'$\bar{r}\,(1/s)$')
plt.savefig('figures/microcircuit_rates.pdf')
plt.show()
