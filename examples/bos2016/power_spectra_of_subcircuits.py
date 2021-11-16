"""
Power Spectra of Sub-Circuits (Bos 2016)
========================================

Here we calculate the power spectra of subcircuits of the :cite:t:`potjans2014` 
microcircuit model including modifications made in :cite:t:`bos2016`.

This example reproduces Fig. 9 in :cite:t:`bos2016`.
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

frequencies = microcircuit.analysis_params['omegas']/(2*np.pi)
full_indegree_matrix = microcircuit.network_params['K']

# %%
# Read the simulated power spectra from the publicated data for comparison.
fix_path = '../../tests/fixtures/integration/data/'
result = nnmt.input_output.load_h5(fix_path + 
                                   'Bos2016_publicated_and_converted_data.h5')

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
# Initialize a new network instance for the reduced circuit explaining the 
# 64 Hz oscillation. First calculate the working point with the full circuit, 
# then reduce the connectivity.

low_gamma_subcircuit = microcircuit.copy()

# construct a matrix with the wanted connections
# "The circuit is composed of the connections corresponding to the five largest 
# matrix elements of Z amp (64 Hz) and the eight largest elements 
# of Z freq (64 Hz)" (Bos 2016)
reducing_matrix = np.zeros((8,8))
reducing_matrix[0,0:4] = 1
reducing_matrix[1,0:2] = 1
reducing_matrix[2:4,2:4] = 1
reducing_matrix[3,0] = 1

low_gamma_subcircuit.network_params.update({
    'K': full_indegree_matrix*reducing_matrix
})

# calculate the transfer function
nnmt.lif.exp.transfer_function(low_gamma_subcircuit, method='taylor')
# calculate the delay distribution matrix
nnmt.network_properties.delay_dist_matrix(low_gamma_subcircuit)
# calculate the effective connectivity matrix
nnmt.lif.exp.effective_connectivity(low_gamma_subcircuit)
# calculate the power spectra
low_gamma_subcircuit_power_spectra = nnmt.lif.exp.power_spectra(
    low_gamma_subcircuit)

# %%
# Initialize a new network instance to calculate results without connections
# from 23E to 4I. First, calculate the working point with the full circuit, 
# then reduce the connectivity.

without_23E_4I = microcircuit.copy()

# construct a matrix with the wanted connections
reducing_matrix = np.ones((8,8))
reducing_matrix[3,0] = 0

without_23E_4I.network_params.update({
    'K': full_indegree_matrix*reducing_matrix
})

# calculate the transfer function
nnmt.lif.exp.transfer_function(without_23E_4I, method='taylor')
# calculate the delay distribution matrix
nnmt.network_properties.delay_dist_matrix(without_23E_4I)
# calculate the effective connectivity matrix
nnmt.lif.exp.effective_connectivity(without_23E_4I)
# calculate the power spectra
without_23E_4I_power_spectra = nnmt.lif.exp.power_spectra(without_23E_4I)

# %%
# Plotting everything together.

# two column figure, 180 mm wide
fig = plt.figure(figsize=(7.08661, 7.08661/2),
                 constrained_layout=True)
grid_specification = gridspec.GridSpec(2, 1, figure=fig)
colormap = 'coolwarm'
labels = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']


gsA = gridspec.GridSpecFromSubplotSpec(1, 3, 
                                        width_ratios=[1, 1, 1], 
                                        subplot_spec=grid_specification[0])

gsB = gridspec.GridSpecFromSubplotSpec(1, 3, 
                                        width_ratios=[1, 1, 1], 
                                        subplot_spec=grid_specification[1])


# reduced circuit for 64 Hz oscillation
for gs, pop_idx in zip(gsA, [0,2,3]):
    ax = fig.add_subplot(gs)

    ax.plot(result['oscillation_origin']['A']['freq_sim'],
                result['oscillation_origin']['A'][f'power{pop_idx}_sim'], 
                color=(0.8, 0.8, 0.8))
    ax.plot(result['oscillation_origin']['A']['freq_sim_av'],
                result['oscillation_origin']['A'][f'power{pop_idx}_sim_av'], 
                color=(0.5, 0.5, 0.5))
    ax.plot(frequencies, power_spectra[:, pop_idx],
            color='black', linestyle='dashed', zorder=2)
    ax.plot(frequencies, low_gamma_subcircuit_power_spectra[:, pop_idx],
            color='black', zorder=10)
    ax.set_yscale('log')

    ax.set_title(labels[pop_idx])
    ax.set_xlabel(r'frequency (1/$s$)')

    ax.set_xticks([20, 40, 60, 80])
    y_minor = matplotlib.ticker.LogLocator(
        base = 10.0, 
        subs = np.arange(1.0, 10.0) * 0.1, 
        numticks = 10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([])
    ax.set_yticks([1e-5,1e-3,1e-1])
    ax.set_ylabel(r'$|C(\omega)|$')
    ax.set_xlim([0.0, 100.0])
    ax.set_ylim([1e-6, 1e0])


# take out 23E -> 4I 
for gs, pop_idx in zip(gsB, [0,2,3]):
    ax = fig.add_subplot(gs)

    ax.plot(result['oscillation_origin']['B']['freq_sim'],
                result['oscillation_origin']['B'][f'power{pop_idx}_sim'], 
                color=(0.8, 0.8, 0.8))
    ax.plot(result['oscillation_origin']['B']['freq_sim_av'],
                result['oscillation_origin']['B'][f'power{pop_idx}_sim_av'], 
                color=(0.5, 0.5, 0.5))
    ax.plot(frequencies, power_spectra[:, pop_idx],
            color='black', linestyle='dashed', zorder=2)
    ax.plot(frequencies, without_23E_4I_power_spectra[:, pop_idx],
            color='black', zorder=10)
    ax.set_yscale('log')

    ax.set_title(labels[pop_idx])
    ax.set_xlabel(r'frequency (1/$s$)')

    ax.set_xticks([20, 40, 60, 80])
    y_minor = matplotlib.ticker.LogLocator(
        base = 10.0, 
        subs = np.arange(1.0, 10.0) * 0.1, 
        numticks = 10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([])
    ax.set_yticks([1e-5,1e-3,1e-1])
    ax.set_ylabel(r'$|C(\omega)|$')
    ax.set_xlim([0.0, 100.0])
    ax.set_ylim([1e-6, 1e0])

plt.savefig('figures/power_spectra_of_subcircuits_Bos2016.png')
