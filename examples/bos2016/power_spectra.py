"""
Power Spectra (Bos 2016)
========================

Here we calculate the power spectra of the :cite:t:`potjans2014` microcircuit
model including modifications made in :cite:t:`bos2016`.

This example reproduces Fig. 1E in :cite:t:`bos2016`.
"""

import nnmt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker

plt.style.use('frontiers.mplstyle')

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
reduce_frequency_resolution = True

if reduce_frequency_resolution:
    microcircuit.change_parameters(changed_analysis_params={'df': 5},
                                overwrite=True)
    derived_analysis_params = (
        microcircuit._calculate_dependent_analysis_parameters())
    microcircuit.analysis_params.update(derived_analysis_params)

# %%
# Calculate all necessary quantities and finally the power spectra.

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

# %%
# Read the simulated power spectra from the publicated data for comparison.

fix_path = '../../tests/fixtures/integration/data/'
result = nnmt.input_output.load_h5(fix_path + 
                                   'Bos2016_publicated_and_converted_data.h5')
simulated_power_spectra_1_window = result['fig_microcircuit']['1']
simulated_power_spectra_20_window = result['fig_microcircuit']['20']

# %%
# Plotting mean-field prediction and simulated results together.

# two column figure, 180 mm wide
fig = plt.figure(figsize=(7.08661, 7.08661/2),
                 constrained_layout=True)
grid_specification = gridspec.GridSpec(2, 4, figure=fig)

for layer in [0, 1, 2, 3]:
    for pop in [0, 1]:
        j = layer*2+pop
        
        ax = fig.add_subplot(grid_specification[pop, layer])
        
        ax.plot(simulated_power_spectra_1_window['freq_sim'],
                   simulated_power_spectra_1_window[f'power{j}'],
                   color=(0.8, 0.8, 0.8),
                   label='simulation')
        ax.plot(simulated_power_spectra_20_window['freq_sim'],
                   simulated_power_spectra_20_window[f'power{j}'], 
                   color=(0.5, 0.5, 0.5),
                   label='simulation avg.')
        ax.plot(microcircuit.analysis_params['omegas']/(2*np.pi),
                   power_spectra[:, j],
                   color='black', zorder=2,
                   label='prediction')
        
        ax.set_xlim([10.0, 400.0])
        ax.set_ylim([1e-6, 1e0])
        ax.set_yscale('log')
        
        population_name = microcircuit.network_params['populations'][j] 
        if population_name == '23E':
            ax.set_title('2/3E')
        elif population_name == '23I':
            ax.set_title('2/3I')
        else:
            ax.set_title(population_name)
            
            
        if pop == 1 and layer == 0:
            ax.set_xlabel(r'frequency $\omega/2\pi\quad(1/\mathrm{s})$')
        else:
            ax.set_xticklabels([])
        
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
            
        if j == 0:
            ax.set_ylabel(r'power spectrum $|C(\omega)|\quad(1/\mathrm{s}^2)$')
            ax.legend()
    
plt.savefig('figures/power_spectra_Bos2016.png')
