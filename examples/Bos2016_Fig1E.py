import nnmt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
import h5py_wrapper.wrapper as h5


plt.style.use('frontiers.mplstyle')

# create network model microcircuit
microcircuit = nnmt.models.Microcircuit(
    network_params='../tests/fixtures/integration/config/Bos2016_network_params.yaml',
    analysis_params='../tests/fixtures/integration/config/Bos2016_analysis_params.yaml')

# %%
# calculate working point for exponentially shape post synaptic currents
nnmt.lif.exp.working_point(microcircuit, method='taylor')
# calculate the transfer function
nnmt.lif.exp.transfer_function(microcircuit, method='taylor')
# calculate the delay distribution matrix
nnmt.network_properties.delay_dist_matrix(microcircuit)
# calculate the effective connectivity matrix
nnmt.lif.exp.effective_connectivity(microcircuit)
# calculate the power spectra
power_spectra = nnmt.lif.exp.power_spectra(microcircuit).T

# %%
fix_path = '../tests/fixtures/integration/data/'
result = h5.load(fix_path + 'Bos2016_publicated_and_converted_data.h5')
# read the simulated power spectra from the publicated data
simulated_power_spectra_1_window = result['fig_microcircuit']['1']
simulated_power_spectra_20_window = result['fig_microcircuit']['20']

# %%
# two column figure, 180 mm wide
fig = plt.figure(figsize=(7.08661, 7.08661/2),
                 constrained_layout=True)
grid_specification = gridspec.GridSpec(2, 4, figure=fig)


for layer in [0, 1, 2, 3]:
    for pop in [0, 1]:
        j = layer*2+pop
        
        ax = fig.add_subplot(grid_specification[pop, layer])
        
        ax.plot(simulated_power_spectra_1_window['freq_sim'],
                   simulated_power_spectra_1_window[f'power{j}'], color=(0.8, 0.8, 0.8))
        ax.plot(simulated_power_spectra_20_window['freq_sim'],
                   simulated_power_spectra_20_window[f'power{j}'], color=(0.5, 0.5, 0.5))
        ax.plot(microcircuit.analysis_params['omegas']/(2*np.pi),
                   power_spectra[j],
                   color='black', zorder=2)
        
        ax.set_xlim([10.0, 400.0])
        ax.set_ylim([1e-6, 1e0])
        ax.set_yscale('log')
        
        ax.set_title(microcircuit.network_params['populations'][j])
        if pop == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r'frequency (1/$s$)')
        
        ax.set_xticks([100, 200, 300])
        y_minor = matplotlib.ticker.LogLocator(
            base = 10.0, 
            subs = np.arange(1.0, 10.0) * 0.1, 
            numticks = 10)
        ax.yaxis.set_minor_locator(y_minor)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks([])
        if j == 0 or j== 1:
            ax.set_yticks([1e-5,1e-3,1e-1])
            ax.set_ylabel(r'$|C(\omega)|$')
    
plt.savefig('Bos2016_Fig1E.png')

# %%