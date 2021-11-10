"""
Microcircuit Firing Rates (Bos 2016)
====================================

Here we calculate the firing rates of the :cite:t:`potjans2014` microcircuit
model including modifications made in :cite:t:`bos2016`.

This example reproduces Fig. 1D in :cite:t:`bos2016`.
"""
# %%
import nnmt
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('frontiers.mplstyle')

# %%
# Create an instance of the network model class `Microcircuit`.
microcircuit = nnmt.models.Microcircuit(
    '../../tests/fixtures/integration/config/Bos2016_network_params.yaml')

# %%
# Calculate the firing rates for exponentially shaped post synaptic currents.
firing_rates = nnmt.lif.exp.firing_rates(microcircuit, method='taylor')

print(f'Mean rates: {firing_rates}')
# %%
# Load the simulated rates publicated in :cite:t:`bos2016` for comparison.
fix_path = '../../tests/fixtures/integration/data/'
result = nnmt.input_output.load_h5(fix_path + 'Bos2016_publicated_and_converted_data.h5')
simulated_rates = result['fig_microcircuit']['rates_sim']*1000

print(f'Mean simulated rates: {simulated_rates}')
# %%
# Plotting 

# one column figure, 85mm wide
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

plt.savefig('figures/microcircuit_firing_rates_Bos2016.pdf')