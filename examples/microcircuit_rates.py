import lif_meanfield_tools as nnmt
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('frontiers.mplstyle')

# create network model microcircuit
microcircuit = nnmt.models.Microcircuit('network_params_microcircuit.yaml')

# calculate firing rates for exponentially shape post synaptic currents
firing_rates = nnmt.lif.exp.firing_rates(microcircuit, method='shift')

print(f'Mean rates: {firing_rates}')

simulated_rates = [0.751, 2.971, 4.29, 6.94, 6.816, 7.698, 1.126, 8.102]
print(f'Mean simulated rates: {simulated_rates}')

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
plt.show()
