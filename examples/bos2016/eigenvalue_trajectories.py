"""
Eigenvalue Trajectories (Bos 2016)
==================================

Here we calculate the eigenvalue trajectories of the :cite:t:`potjans2014` 
microcircuit model including modifications made in :cite:t:`bos2016`.

This example reproduces Fig 4. in :cite:t:`bos2016`.
"""

import nnmt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict


plt.style.use('frontiers.mplstyle')

# %%
# Create an instance of the network model class `Microcircuit`.
microcircuit = nnmt.models.Microcircuit(
    network_params='../../tests/fixtures/integration/config/Bos2016_network_params.yaml',
    analysis_params='../../tests/fixtures/integration/config/Bos2016_analysis_params.yaml')
frequencies = microcircuit.analysis_params['omegas']/(2.*np.pi)
full_indegree_matrix = microcircuit.network_params['K']

# %%
# Calculate all necessary quantities and finally the eigenvalues of the 
# effective connectivity matrix, sensitivity dictionary and the power spectra.

# calculate working point for exponentially shape post synaptic currents
nnmt.lif.exp.working_point(microcircuit, method='taylor')
# calculate the transfer function
nnmt.lif.exp.transfer_function(microcircuit, method='taylor')
# calculate the delay distribution matrix
nnmt.network_properties.delay_dist_matrix(microcircuit)
eigenvalues = np.linalg.eig(nnmt.lif.exp.effective_connectivity(microcircuit))[0]
resorted_eigenvalues, new_indices = nnmt.lif.exp._resort_eigenvalues(eigenvalues)
sensitivity_dict = nnmt.lif.exp.sensitivity_measure_per_eigenmode(network=microcircuit)
# calculate the power spectra
power_spectra = nnmt.lif.exp.power_spectra(microcircuit).T

# %%
# Identify the indices of the eigenvalue that should be plotted to 
# reproduce Fig. 4 in Bos 2016.

for i in range(8):
    print(sensitivity_dict[i]['critical_frequency'])
    
# manually identified the following indices to correspond to the markers shown
# in :cite:t:`bos2016`
eigenvalues_to_be_plotted = [0, 1, 2, 3, 5, 6]
print(f'Eigenvalues to be plotted: {eigenvalues_to_be_plotted}')

# %%
# Alter the indegree matrix to consist just of the individual isolated layers
# and calculate the corresponding eigenvalue spectra and power spectra.

isolated_layers_results = defaultdict(str)
for i, layer in enumerate(['23', '4', '5', '6']):
    print(f'Modify connectivity to obtain isolated {layer}.')
    isolated_layers_results[layer] = defaultdict(str)
    
    microcircuit_isolated_layers = nnmt.models.Microcircuit(
        network_params='../../tests/fixtures/integration/config/Bos2016_network_params.yaml',
        analysis_params='../../tests/fixtures/integration/config/Bos2016_analysis_params.yaml')
    isolated_layers_results[layer]['network'] = microcircuit_isolated_layers
    
    # calculate working point for exponentially shape post synaptic currents
    nnmt.lif.exp.working_point(isolated_layers_results[layer]['network'], method='taylor')

    reducing_matrix = np.zeros((8,8))
    reducing_matrix[2*i:2*i+2, 2*i:2*i+2] = np.ones([2,2])
    # for layer i, set indegree matrix such that it is isolated
    isolated_layers_results[layer]['network'].network_params.update(
        K = full_indegree_matrix*reducing_matrix)
    
    print('connectivity changed...')
    # calculate the transfer function
    nnmt.lif.exp.transfer_function(isolated_layers_results[layer]['network'], method='taylor')
    # calculate the delay distribution matrix
    nnmt.network_properties.delay_dist_matrix(isolated_layers_results[layer]['network'])
    eigenvalue_spectra_layer = np.linalg.eig(
            nnmt.lif.exp.effective_connectivity(isolated_layers_results[layer]['network']))[0].T
    # calculate the power spectra
    power_spectra_layer = nnmt.lif.exp.power_spectra(isolated_layers_results[layer]['network']).T
    
    resorted_eigenvalue_spectra_layer, new_indices_layer = nnmt.lif.exp._resort_eigenvalues(
        eigenvalue_spectra_layer)
    
    isolated_layers_results[layer]['eigenvalue_spectra'] = eigenvalue_spectra_layer
    isolated_layers_results[layer]['power_spectra'] = power_spectra_layer  

# %%
# Calculate the sensitivity measure dictionary for each isolated layer.
for i, layer in enumerate(['23', '4', '5', '6']):
    layer = isolated_layers_results[layer]
    layer['sensitivity_dict'] = nnmt.lif.exp.sensitivity_measure_per_eigenmode(layer['network'])

# identify which eigenvalues should be plotted to reproduce Fig.4 of Bos 2016
for i, layer in enumerate(['23', '4', '5', '6']):
    for j in range(2):
        print(layer, j, isolated_layers_results[layer]['sensitivity_dict'][j]['critical_frequency'])

layer_eigenvalue_tuples_to_be_plotted = [('23', 1), ('4', 1),
                                         ('5', 1), ('6', 1),
                                         ('23', 0), ('5', 0)]

# %%
# Plotting

# two column figure, 180 mm wide
fig = plt.figure(figsize=(7.08661, 7.08661/2),
                 constrained_layout=True)
grid_specification = gridspec.GridSpec(1, 3, figure=fig)

# Panel A
gsA = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_specification[0])
# Panel B
gsB = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_specification[1])
# Panel C
gsC = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_specification[2])

### Panel A
# top
ax = fig.add_subplot(gsA[0])
N = resorted_eigenvalues.shape[0]
dc = 1/float(N)
for i in range(0, N, 3):
    ax.plot(resorted_eigenvalues[i].real, 
            resorted_eigenvalues[i].imag, '.',
                color=(0.9-0.9*i*dc, 0.9-0.9*i*dc, 0.9-0.9*i*dc),
                markersize=1.0, zorder=1)
ax.scatter(1,0, s=15, color='r')
ax.set_ylim([-4, 6.5])
ax.set_xlim([-11.5, 2])
ax.set_xticks([-9, -6, -3, 0])
ax.set_yticks([-3, 0, 3, 6])
ax.set_ylabel('Im($\lambda(\omega)$)')

# bottom
ax = fig.add_subplot(gsA[1])
for i in range(0, N, 3):
    ax.plot(resorted_eigenvalues[i].real, 
            resorted_eigenvalues[i].imag, '.',
                color=(0.9-0.9*i*dc, 0.9-0.9*i*dc, 0.9-0.9*i*dc),
                markersize=1.0, zorder=1)
# frequencies where eigenvalue trajectory is closest to one
fmaxs = [np.round(sensitivity_dict[i]['critical_frequency'],1) for i in eigenvalues_to_be_plotted]
markers = ['<', '>', '^', 'v', 'o', '+']
for i, eig_index in enumerate(eigenvalues_to_be_plotted):
    eigc = sensitivity_dict[eig_index]['critical_eigenvalue']
    ax.plot(eigc.real, eigc.imag, markers[i], color='black',#colors_array[i],
                mew=1, ms=4, label=str(fmaxs[i])+'Hz')
ax.legend(bbox_to_anchor=(-0.35, -0.9, 1.6, 0.5), loc='center', 
                ncol=3, mode="expand", borderaxespad=3.5, fontsize=7,
                numpoints = 1)
ax.scatter(1,0, s=5, color='r')
# box = ax.get_position()
# ax.set_position([box.x0, box.y0-box.height, box.width, box.height*2])
ax.set_xlabel('Re($\lambda(\omega)$)')
ax.set_ylabel('Im($\lambda(\omega)$)')
ax.set_ylim([-0.3, 0.5])
ax.set_xlim([0.1, 1.1])
ax.set_yticks([-0.2, 0, 0.2, 0.4])
ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])


### Panel B

def get_parameter_plot():
    colors = [[] for i in range(4)]
    colors[1] = [(0.0, 0.7, 0.0), (0.0, 1.0, 0.0)]
    colors[3] = [(0.0, 0.0, 0.4), (0.0, 0.0, 1.0)]
    colors[0] = [(0.7, 0.0, 0.0), (1.0, 0.0, 0.0)]
    colors[2] = [(0.5, 0.0, 0.5), (1.0, 0.0, 1.0)]
    return colors

colors = get_parameter_plot()

def get_color(i, layer):
    cont_colors = [(1.0-i*dc, 0.0, 0.0), (0.0, 1.0-i*dc, 0.0), 
                    (1.0-i*dc, 0.0, 1.0-i*dc), (0.0, 0.0, 1.0-i*dc)]
    index = ['23', '4', '5',  '6'].index(layer)
    return cont_colors[index]

# top
ax = fig.add_subplot(gsB[0])
N = resorted_eigenvalues.shape[0]
dc = 1/float(N)
for i, layer in enumerate(['23', '4', '5', '6']):
    # [:, 2*layer:2*layer+2] in the original code serves to plot only the non-zero eigenspectra 
    for j in range(N):
        plt.scatter(isolated_layers_results[layer]['eigenvalue_spectra'][:, j].real,
                    isolated_layers_results[layer]['eigenvalue_spectra'][:, j].imag,
                    color=get_color(j, layer),
                    s=0.5, zorder=1)
ax.scatter(1,0, s=15, color='r')
ax.set_ylim([-4, 6.5])
ax.set_xlim([-11.5, 2])
ax.set_xticks([-9, -6, -3, 0])
ax.set_yticks([-3, 0, 3, 6])
ax.set_ylabel('Im($\lambda(\omega)$)')

# bottom
ax = fig.add_subplot(gsB[1])
for i, layer in enumerate(['23', '4', '5', '6']):
    for j in range(N):
        plt.scatter(isolated_layers_results[layer]['eigenvalue_spectra'][:, j].real,
                    isolated_layers_results[layer]['eigenvalue_spectra'][:, j].imag,
                    color=get_color(j, layer),
                    s=0.5, zorder=1)
        
# frequencies where eigenvalue trajectory is closest to one
fmaxs = [np.round(isolated_layers_results[i[0]]['sensitivity_dict'][i[1]]['critical_frequency'],1) for i in layer_eigenvalue_tuples_to_be_plotted]
markers = ['<', '>', '^', 'v', 'o', '+']
for i, (layer, eig_index) in enumerate(layer_eigenvalue_tuples_to_be_plotted):
    eigc = isolated_layers_results[layer]['sensitivity_dict'][eig_index]['critical_eigenvalue']
    ax.plot(eigc.real, eigc.imag, markers[i], color='black',#colors_array[i],
                mew=1, ms=4, label=str(fmaxs[i])+'Hz')
ax.legend(bbox_to_anchor=(-0.35, -0.9, 1.6, 0.5), loc='center', 
                ncol=3, mode="expand", borderaxespad=3.5, fontsize=7,
                numpoints = 1)
ax.scatter(1,0, s=5, color='r')
ax.set_xlabel('Re($\lambda(\omega)$)')
ax.set_ylabel('Im($\lambda(\omega)$)')
ax.set_ylim([-0.3, 0.5])
ax.set_xlim([0.1, 1.1])
ax.set_yticks([-0.2, 0, 0.2, 0.4])
ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])

### panel C ##
freqs_isolated_layers = microcircuit_isolated_layers.analysis_params['omegas']/(2*np.pi)
frequencies = microcircuit.analysis_params['omegas']/(2*np.pi)

# loop across layers 23 and 4
for i, layer in enumerate(['23', '4']):
    ax = fig.add_subplot(gsC[i])

    # loop across excitatory and inhibitory
    for j in [0,1]:
        ax.plot(freqs_isolated_layers, isolated_layers_results[layer]['power_spectra'][j+2*i], 
                        color=colors[i][j])
        ax.plot(frequencies, power_spectra[2*i+j], 
                        color='black', linestyle='dashed')
        
    ax.set_yscale('log')
    ax.set_xticks([100, 200, 300])
    ax.set_ylabel('power')
    ax.set_yticks([1e-2, 1e-4])

    if i == 0:
        ax.set_ylim([5*1e-6, 5*1e-2])
    else:
        ax.set_ylim([2*1e-5, 4*1e-1])
        ax.set_xlabel('frequency $f$(1/$s$)')
        
plt.savefig('figures/eigenvalue_trajectories_Bos2016.pdf')