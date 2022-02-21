"""
Sensitivity Measure (Bos 2016)
==============================

Here we calculate the sensitivity measure of the :cite:t:`potjans2014` 
microcircuit model including modifications made in :cite:t:`bos2016`.

This example reproduces parts of Fig. 5 and 8 in :cite:t:`bos2016`.
"""

import nnmt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import matplotlib.ticker


def colorbar(mappable, cax=None):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    if cax==None:
        cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

plt.style.use('frontiers.mplstyle')
mpl.rcParams.update({'legend.fontsize': 'medium',  # old: 5.0 was too small
                     'axes.titlepad': 0.0})
# %% 
# First, create an instance of the network model class `Microcircuit`.
microcircuit = nnmt.models.Microcircuit(
    network_params=
    '../../tests/fixtures/integration/config/Bos2016_network_params.yaml',
    analysis_params=
    '../../tests/fixtures/integration/config/Bos2016_analysis_params.yaml')

# %%
# The frequency resolution used in the original publication was quite high.
# Here, we reduce the frequency resolution for faster execution.
reduce_frequency_resolution = False

if reduce_frequency_resolution:
    microcircuit.change_parameters(changed_analysis_params={'df': 1},
                                overwrite=True)
    derived_analysis_params = (
        microcircuit._calculate_dependent_analysis_parameters())
    microcircuit.analysis_params.update(derived_analysis_params)

frequencies = microcircuit.analysis_params['omegas']/(2.*np.pi)
# %%
# Calculate all necessary quantities and finally the sensitivity 
# measure for all eigenmodes.

# calculate working point for exponentially shape post synaptic currents
nnmt.lif.exp.working_point(microcircuit, method='taylor')
# calculate the transfer function
nnmt.lif.exp.transfer_function(microcircuit, method='taylor')
# calculate the delay distribution matrix
nnmt.network_properties.delay_dist_matrix(microcircuit)
# calculate the effective connectivity matrix
nnmt.lif.exp.effective_connectivity(microcircuit)
# calculate the power spectra
power_spectra = nnmt.lif.exp.power_spectra(microcircuit)

sensitivity_dict = nnmt.lif.exp.sensitivity_measure_all_eigenmodes(
    microcircuit)

# %%
g =  microcircuit.network_params['g']
population_rate = microcircuit.results['lif.exp.firing_rates'][3]
external_input_rate = microcircuit.network_params['nu_ext']
connections_4I_4I = microcircuit.network_params['K'][3,3]
connection_external_4I = microcircuit.network_params['K_ext'][3]
# %%
# in order to keep the working point constant when changing one connection,
# we modify the connections from the external input to the target population
# (use Eq. 2 in Bos manuscript for this)

percentage_of_change = 0.05

external_connections_to_add = (
    abs(g) * connections_4I_4I * percentage_of_change * population_rate
    ) / external_input_rate

K_5_percent = microcircuit.network_params['K'].copy()
K_5_percent_ext = microcircuit.network_params['K_ext'].copy()
K_5_percent[3,3] = K_5_percent[3,3]*(1+percentage_of_change)
K_5_percent_ext[3] = K_5_percent_ext[3] + external_connections_to_add 

microcircuit_5_percent = microcircuit.change_parameters(
    {'K': K_5_percent,
     'K_ext': K_5_percent_ext})

# Calculate all necessary quantities and finally the power spectra.

# calculate working point for exponentially shape post synaptic currents
nnmt.lif.exp.working_point(microcircuit_5_percent, method='taylor')
# calculate the transfer function
nnmt.lif.exp.transfer_function(microcircuit_5_percent, method='taylor')
# calculate the delay distribution matrix
nnmt.network_properties.delay_dist_matrix(microcircuit_5_percent)
# calculate the effective connectivity matrix
nnmt.lif.exp.effective_connectivity(microcircuit_5_percent)
# calculate the power spectra
power_spectra_5_percent = nnmt.lif.exp.power_spectra(microcircuit_5_percent)

# %%
# in order to keep the working point constant when changing one connection,
# we modify the connections from the external input to the target population
# (use Eq. 2 in Bos manuscript for this)

percentage_of_change = 0.10

external_connections_to_add = (
    abs(g) * connections_4I_4I * percentage_of_change * population_rate
    ) / external_input_rate

K_10_percent = microcircuit.network_params['K'].copy()
K_10_percent_ext = microcircuit.network_params['K_ext'].copy()
K_10_percent[3,3] = K_10_percent[3,3]*(1+percentage_of_change)
K_10_percent_ext[3] = K_10_percent_ext[3] + external_connections_to_add 

microcircuit_10_percent = microcircuit.change_parameters(
    {'K': K_10_percent,
     'K_ext': K_10_percent_ext})

# Calculate all necessary quantities and finally the power spectra.

# calculate working point for exponentially shape post synaptic currents
nnmt.lif.exp.working_point(microcircuit_10_percent, method='taylor')
# calculate the transfer function
nnmt.lif.exp.transfer_function(microcircuit_10_percent, method='taylor')
# calculate the delay distribution matrix
nnmt.network_properties.delay_dist_matrix(microcircuit_10_percent)
# calculate the effective connectivity matrix
nnmt.lif.exp.effective_connectivity(microcircuit_10_percent)
# calculate the power spectra
power_spectra_10_percent = nnmt.lif.exp.power_spectra(microcircuit_10_percent)
# %%

# Look at the critical frequencies per eigenmode
for k, v in sensitivity_dict.items():
    print(k, v['critical_frequency'])
# %%
# two column figure, 180 mm wide
width = 180. / 25.4 
height = 80. / 25.4

fig = plt.figure(figsize=(width, height),
                 constrained_layout=True)

labels = ['2/3E', '2/3I', '4E', '4I', '5E', '5I', '6E', '6I']

colormap = mpl.cm.get_cmap('coolwarm').copy()

# set colorbar max and min
z = 1

# by default np.nan are set to black with full transparency
# .eps can't handle transparency
colormap.set_bad('w',1.)


fig = plt.figure(figsize=(width, height),
                 constrained_layout=True)
grid_specification = gridspec.GridSpec(1, 2, 
                                       height_ratios=[1],
                                       width_ratios=[2.2, 1], figure=fig)

gs = gridspec.GridSpecFromSubplotSpec(1,3, 
                                        height_ratios=[1],
                            width_ratios=[1, 1, 0.2], 
                            subplot_spec=grid_specification[0])


ev = '5'
ax = fig.add_subplot(gs[0])
label_prms = dict(x=-0.3, y=1.2, fontsize=10, fontweight='bold',
                  va='top', ha='right')

ax.text(s='(A)', transform=ax.transAxes, **label_prms)

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

plot_title = r'$\mathbf{Z}_{j=%s}^{\mathrm{freq}}(' % ev + \
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

if labels is not None:
    ax.set_xticks(np.arange(len(labels))+0.5)
    ax.set_yticks(np.arange(len(labels))+0.5)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels([])

ax.set_xlabel('sources')
    
colorbar_ax = fig.add_subplot(gs[2])

colorbar_width = 0.1
ip = InsetPosition(ax, [1.05,0,colorbar_width,1]) 
colorbar_ax.set_axes_locator(ip)
colorbar(heatmap, cax=colorbar_ax)


ax = fig.add_subplot(grid_specification[1])
label_prms = dict(x=-0.2, y=1.0, fontsize=10, fontweight='bold',
                  va='top', ha='right')
ax.text(s='(B)', transform=ax.transAxes, **label_prms)
j = 3
ax.plot(microcircuit.analysis_params['omegas']/(2*np.pi),
            power_spectra[:, j],
            color='k', zorder=2,
            label=str(
                r'$ K_{\mathrm{4I}\rightarrow\mathrm{4I}}$ = ' +
                f"{int(np.round(microcircuit.network_params['K'][3][3],0))}" +
                r'$, K_{\mathrm{ext}\rightarrow\mathrm{4I}}$ = ' +
                f"{np.round(microcircuit.network_params['K_ext'][3],0)}"
                )
            )
                

ax.plot(microcircuit_5_percent.analysis_params['omegas']/(2*np.pi),
            power_spectra_5_percent[:, j],
            color='k', alpha=0.5, zorder=2, ls='dashed',
            label=str(
                r'$ K_{\mathrm{4I}\rightarrow\mathrm{4I}}$ = ' +
                f"{int(np.round(microcircuit_5_percent.network_params['K'][3][3],0))}" +
                r'$, K_{\mathrm{ext}\rightarrow\mathrm{4I}}$ = ' +
                f"{np.round(microcircuit_5_percent.network_params['K_ext'][3],0)}"
                )
            )

ax.plot(microcircuit_10_percent.analysis_params['omegas']/(2*np.pi),
    power_spectra_10_percent[:, j],
    color='k', alpha=0.2, zorder=2, ls='dotted',
            label=str(
                r'$ K_{\mathrm{4I}\rightarrow\mathrm{4I}}$ = ' +
                f"{int(np.round(microcircuit_10_percent.network_params['K'][3][3],0))}" +
                r'$, K_{\mathrm{ext}\rightarrow\mathrm{4I}}$ = ' +
                f"{np.round(microcircuit_10_percent.network_params['K_ext'][3],0)}"
                )
            )

ax.set_xlim([0.0, 100.0])
ax.set_ylim([1e-6, 1e0])
ax.set_yscale('log')
ax.set_xticks([20, 40, 60, 80])

ax.set_xlabel(r'frequency $\omega/2\pi\quad(1/\mathrm{s})$')

y_minor = matplotlib.ticker.LogLocator(
    base = 10.0, 
    subs = np.arange(1.0, 10.0) * 0.1, 
    numticks = 10)
ax.yaxis.set_minor_locator(y_minor)
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks([])
ax.set_yticks([1e-5,1e-3,1e-1])
ax.set_ylabel(r'power spectrum $P(\omega_{\mathrm{4I}})\quad(1/\mathrm{s}^2)$')
ax.legend(loc='lower right')

# plt.savefig('figures/sensitivity_measure_low_gamma_Bos2016.eps')

# %%
