# %%
from nnmt.models import network
import nnmt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
import h5py_wrapper.wrapper as h5
from collections import defaultdict


from mpl_toolkits.axes_grid1 import make_axes_locatable

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

plt.style.use('frontiers.mplstyle')

# create network model microcircuit
microcircuit = nnmt.models.Microcircuit(
    network_params=\
        '../tests/fixtures/integration/config/Bos2016_network_params.yaml',
    analysis_params=\
        '../tests/fixtures/integration/config/Bos2016_analysis_params.yaml')

frequencies = microcircuit.analysis_params['omegas']/(2.*np.pi)

# calculate working point for exponentially shape post synaptic currents
nnmt.lif.exp.working_point(microcircuit, method='taylor')
# calculate the transfer function
nnmt.lif.exp.transfer_function(microcircuit, method='taylor')
# calculate the delay distribution matrix
nnmt.network_properties.delay_dist_matrix(microcircuit)

sensitivity_dict = nnmt.lif.exp.sensitivity_measure_dictionary(microcircuit)

# print necessary entries of the sensitivity measure dictionary to see
# which eigenvalues are needed to reproduce Fig.6 and Fig.7 of Bos 2016
for i in range(8):
    print(sensitivity_dict[i]['critical_frequency'])
    print(sensitivity_dict[i]['critical_eigenvalue'])
    print(sensitivity_dict[i]['k'])
    print(sensitivity_dict[i]['k_per'])    
    
eigenvalues_to_plot_high = [1, 0, 3, 2]
eigenvalue_to_plot_low = 6

# plot Fig. 6
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
    
    # senstivity_measure_amplitude
    ax = fig.add_subplot(gs[0])
    
    frequency = sensitivity_dict[ev]['critical_frequency']
    projection_of_sensitivity_measure = sensitivity_dict[ev][
        'sensitivity_measure_amp']
    
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
    
    # senstivity_measure_frequency
    ax = fig.add_subplot(gs[1])
    
    frequency = sensitivity_dict[ev]['critical_frequency']
    projection_of_sensitivity_measure = sensitivity_dict[ev][
        'sensitivity_measure_freq']
    
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
    
plt.savefig('Bos2016_Fig6.png', bbox_inches='tight')

# plot Fig. 7
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
    'sensitivity_measure_amp']

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

plt.savefig('Bos2016_Fig7.png', bbox_inches='tight')