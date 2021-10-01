"""
Sensitivity Measure (Bos 2016)
==============================

Here we calculate the sensitivity measure of the :cite:t:`potjans2014` 
microcircuit model including modifications made in :cite:t:`bos2016`.

This example reproduces Fig. 6 and 7 in :cite:t:`bos2016`.
"""

import nnmt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

plt.style.use('frontiers.mplstyle')

# %% 
# Create an instance of the network model class `Microcircuit`.
microcircuit = nnmt.models.Microcircuit(
    network_params='../tests/fixtures/integration/config/Bos2016_network_params.yaml',
    analysis_params='../tests/fixtures/integration/config/Bos2016_analysis_params.yaml')
frequencies = microcircuit.analysis_params['omegas']/(2.*np.pi)

# %%
# Calculate all necessary quantities and finally the sensitivity 
# measure dictionary.

# calculate working point for exponentially shape post synaptic currents
nnmt.lif.exp.working_point(microcircuit, method='taylor')
# calculate the transfer function
nnmt.lif.exp.transfer_function(microcircuit, method='taylor')
# calculate the delay distribution matrix
nnmt.network_properties.delay_dist_matrix(microcircuit)
# calculate the effective connectivity matrix
nnmt.lif.exp.effective_connectivity(microcircuit)

sensitivity_dict = nnmt.lif.exp.sensitivity_measure_per_eigenmode(microcircuit)

# print necessary entries of the sensitivity measure dictionary to see
# which eigenvalues are needed to reproduce Fig.6 and Fig.7 of Bos 2016
for i in range(8):
    print(sensitivity_dict[i]['critical_frequency'])
    print(sensitivity_dict[i]['critical_eigenvalue'])
    print(sensitivity_dict[i]['k'])
    print(sensitivity_dict[i]['k_per'])    
    
# identified these indices manually    
eigenvalues_to_plot_high = [1, 0, 3, 2]
eigenvalue_to_plot_low = 6

# %%
# Plotting: Sensitivity Measure corresponding to high frequency peak (Fig. 6)

# two column figure, 180 mm wide
fig = plt.figure(figsize=(7.08661, 7.08661/2),
                 constrained_layout=True)
grid_specification = gridspec.GridSpec(2, 2, figure=fig)
colormap = 'coolwarm'
labels = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']
plt.rcParams['xtick.labelsize'] = 'x-small'
plt.rcParams['ytick.labelsize'] = 'x-small'

for ev, subpanel in zip(eigenvalues_to_plot_high, grid_specification):
    gs = gridspec.GridSpecFromSubplotSpec(1, 2, 
                                           width_ratios=[1, 1], 
                                           subplot_spec=subpanel)
    
    # sensitivity_measure_amplitude
    ax = fig.add_subplot(gs[0])
    
    frequency = sensitivity_dict[ev]['critical_frequency']
    projection_of_sensitivity_measure = sensitivity_dict[ev][
        'sensitivity_amp']
    
    # obtain maximal absolute value
    z = np.max(abs(projection_of_sensitivity_measure))
    rounded_frequency = str(np.round(frequency,1))

    plot_title = r'$\mathbf{Z}^{\mathrm{amp}}(' + \
        f'{rounded_frequency}' + r'\mathrm{Hz})$'
    ax.set_title(plot_title)

    heatmap = ax.imshow(projection_of_sensitivity_measure,
                        vmin=-z,
                        vmax=z,
                        cmap=colormap)
    colorbar(heatmap)
    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

    ax.set_xlabel('sources')
    ax.set_ylabel('targets')
    
    # sensitivity_measure_frequency
    ax = fig.add_subplot(gs[1])
    
    frequency = sensitivity_dict[ev]['critical_frequency']
    projection_of_sensitivity_measure = sensitivity_dict[ev][
        'sensitivity_freq']
    
    # obtain maximal absolute value
    z = np.max(abs(projection_of_sensitivity_measure))
    rounded_frequency = str(np.round(frequency,1))

    plot_title = r'$\mathbf{Z}^{\mathrm{freq}}(' + \
        f'{rounded_frequency}' + r'\mathrm{Hz})$'
    ax.set_title(plot_title)

    heatmap = ax.imshow(projection_of_sensitivity_measure,
                        vmin=-z,
                        vmax=z,
                        cmap=colormap)
    colorbar(heatmap)
    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

    ax.set_xlabel('sources')
    ax.set_ylabel('targets')

fig.set_constrained_layout_pads(w_pad=2 / 72, h_pad=2 / 72, hspace=0.1,
                                wspace=0.1)    
    
plt.savefig('figures/sensitivity_measure_high_gamma_Bos2016.pdf', bbox_inches='tight')

# %%
# Plotting: Sensitivity Measure corresponding to low frequencies (Fig. 7)

# two column figure, 180 mm wide
fig = plt.figure(figsize=(3.34646, 3.34646/2),
                 constrained_layout=True)
colormap = 'coolwarm'
labels = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']
plt.rcParams['xtick.labelsize'] = 'x-small'
plt.rcParams['ytick.labelsize'] = 'x-small'

ax = fig.add_subplot(111)

ev = eigenvalue_to_plot_low

frequency = sensitivity_dict[ev]['critical_frequency']
projection_of_sensitivity_measure = sensitivity_dict[ev][
    'sensitivity_amp']

# obtain maximal absolute value
z = np.max(abs(projection_of_sensitivity_measure))
rounded_frequency = str(np.round(frequency,1))

plot_title = r'$\mathbf{Z}^{\mathrm{amp}}(' + \
        f'{rounded_frequency}' + r'\mathrm{Hz})$'
ax.set_title(plot_title)

heatmap = ax.imshow(projection_of_sensitivity_measure,
                    vmin=-z,
                    vmax=z,
                    cmap=colormap)
colorbar(heatmap)
if labels is not None:
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

ax.set_xlabel('sources')
ax.set_ylabel('targets')

plt.savefig('figures/sensitivity_measure_low_gamma_Bos2016.pdf', bbox_inches='tight')