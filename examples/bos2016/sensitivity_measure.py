"""
Sensitivity Measure (Bos 2016)
==============================

Here we calculate the sensitivity measure of the :cite:t:`potjans2014` 
microcircuit model including modifications made in :cite:t:`bos2016`.

This example reproduces Fig. 6 and 7 in :cite:t:`bos2016`.
"""

import nnmt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition


def colorbar(mappable, cax=None):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    if cax==None:
        cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

plt.style.use('frontiers.mplstyle')

# %% 
# Create an instance of the network model class `Microcircuit`.
microcircuit = nnmt.models.Microcircuit(
    network_params='../../tests/fixtures/integration/config/Bos2016_network_params.yaml',
    analysis_params='../../tests/fixtures/integration/config/Bos2016_analysis_params.yaml')
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

sensitivity_dict = nnmt.lif.exp.sensitivity_measure_all_eigenmodes(microcircuit)
# %%
import pickle
# pickle.dump(sensitivity_dict, open('sensitivity_dict.pkl', 'wb'))
sensitivity_dict = pickle.load(open('sensitivity_dict.pkl', 'rb'))
# %%
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
# grid_specification.update(
#     left=0.05, right=0.95, bottom=0.08, top=0.93, wspace=0.05, hspace=0.1)

labels = ['2/3E', '2/3I', '4E', '4I', '5E', '5I', '6E', '6I']

# too small for frontiers
# plt.rcParams['xtick.labelsize'] = 'x-small'
# plt.rcParams['ytick.labelsize'] = 'x-small'

colormap = mpl.cm.get_cmap('coolwarm').copy()
# set colorbar max and min
z = 1

# by default np.nan are set to black with full transparency
# .eps can't handle transparency
colormap.set_bad('w',1.)

label_prms = dict(x=-0.3, y=1.2, fontsize=10, fontweight='bold',
                  va='top', ha='right')
panel_labels = ['(A)', '(B)', '(C)', '(D)']

for count, (ev, subpanel, panel_label) in enumerate(
    zip(eigenvalues_to_plot_high, grid_specification, panel_labels)):

    gs = gridspec.GridSpecFromSubplotSpec(1,3, 
                                          height_ratios=[1],
                                width_ratios=[1, 1, 0.2], 
                                subplot_spec=subpanel)

    # sensitivity_measure_amplitude
    ax = fig.add_subplot(gs[0])
    ax.text(s=panel_label, transform=ax.transAxes, **label_prms)
    
    frequency = sensitivity_dict[ev]['critical_frequency']
    projection_of_sensitivity_measure = sensitivity_dict[ev][
        'sensitivity_amp']
    
    rounded_frequency = str(int(np.round(frequency,0)))

    plot_title = r'$\mathbf{Z}_{j=%s}^{\mathrm{amp}}(' % ev + \
        f'{rounded_frequency}' + r'\,\mathrm{Hz})$'
    ax.set_title(plot_title)

    data = np.ma.masked_where(projection_of_sensitivity_measure == 0,
                              projection_of_sensitivity_measure)

    # np.flipud needed to cope for different origin compared to imshow
    heatmap = ax.pcolormesh(np.flipud(data),
                        vmin=-z,
                        vmax=z,
                        cmap=colormap,
                        edgecolors='k',
                        linewidth=0.6)

    
    ax.set_aspect('equal')

    # Minor ticks
    # ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    # ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    # ax.grid(which='minor', color='k', linestyle='-', linewidth=0.6)
    # ax.tick_params(axis='x', which='minor', bottom=False)
    # ax.tick_params(axis='y', which='minor', left=False)

    
    if labels is not None:
        ax.set_xticks(np.arange(len(labels))+0.5)
        ax.set_yticks(np.arange(len(labels))+0.5)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(list(reversed(labels)))

    ax.set_xlabel('sources')
    ax.set_ylabel('targets')
    

    # sensitivity_measure_frequency
    ax = fig.add_subplot(gs[1])
    
    frequency = sensitivity_dict[ev]['critical_frequency']
    projection_of_sensitivity_measure = sensitivity_dict[ev][
        'sensitivity_freq']
    
    rounded_frequency = str(int(np.round(frequency,0)))

    plot_title = r'$\mathbf{Z}_{j=%s}^{\mathrm{amp}}(' % ev + \
        f'{rounded_frequency}' + r'\,\mathrm{Hz})$'
    ax.set_title(plot_title)
    
    data = np.ma.masked_where(projection_of_sensitivity_measure == 0,
                              projection_of_sensitivity_measure)

    heatmap = ax.pcolormesh(np.flipud(data),
                        vmin=-z,
                        vmax=z,
                        cmap=colormap,    
                        edgecolors='k',
                        linewidth=0.6)
    
    ax.set_aspect('equal')
    
    
    # Minor ticks
    # ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    # ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    # ax.grid(which='minor', color='k', linestyle='-', linewidth=0.6)
    # ax.tick_params(axis='x', which='minor', bottom=False)
    # ax.tick_params(axis='y', which='minor', left=False)
    
    # ax.grid(True, color='k', linestyle='-', linewidth=0.6)
    
    if labels is not None:
        ax.set_xticks(np.arange(len(labels))+0.5)
        ax.set_yticks(np.arange(len(labels))+0.5)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels([])

    ax.set_xlabel('sources')
    # ax.set_ylabel('targets')
        
    colorbar_ax = fig.add_subplot(gs[2])
    
    colorbar_width = 0.1
    ip = InsetPosition(ax, [1.05,0,colorbar_width,1]) 
    colorbar_ax.set_axes_locator(ip)
    colorbar(heatmap, cax=colorbar_ax)

fig.set_constrained_layout_pads(w_pad=0, h_pad=0,
                                hspace=0.1, wspace=0.1)    
    
plt.savefig('figures/sensitivity_measure_high_gamma_Bos2016.eps')

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
rounded_frequency = str(np.round(frequency,2))

plot_title = r'$\mathbf{Z}^{\mathrm{amp}}(' + \
        f'{rounded_frequency}' + r'\,\mathrm{Hz})$'
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
# %%
