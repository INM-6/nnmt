import nnmt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
import h5py_wrapper.wrapper as h5
from collections import defaultdict


plt.style.use('frontiers.mplstyle')

# %%
def calculate_distance_in_complex_plane(a, b):
    """
    a and b are complex numbers
    """
    distance = np.sqrt((b.real - a.real)**2 + (b.imag - a.imag)**2)
    return distance

def resort_eigenvalues(eigenvalues, margin=1e-5):
    """
    The eigenvalues of the effective connectivity are calculated per frequency.
    To link the eigenvalues/eigenmodes across frequencies this utility function
    calculates the distance between subsequent (in frequency) eigenvalues and matches
    them if the distance is smaller equal the margin.
    """
    
    eig = eigenvalues.copy()
    
    # define vector of eigenvalue at frequency 0
    previous = eig[:, 0]
    
    # initialize containers
    distances = np.zeros([eig.shape[0], eig.shape[1] - 1])
    multi_swaps = {}
    new_indices = np.tile(np.arange(eig.shape[0]), (eig.shape[1], 1)).T
    
    # loop over all frequences > 0
    for i in range(1, eig.shape[1]):
        # compare new to previous
        new = eig[:, i]
        distances[:, i-1] = calculate_distance_in_complex_plane(previous, new)

        # get all distances which are larger then margin
        if np.any(distances[:, i-1] > margin):
            indices = np.argwhere(distances[:, i-1] > margin).reshape(-1)
            if len(indices) >= 2:
                multi_swaps[i-1] = indices
        previous = new

    if multi_swaps:
        for n, (i, j) in enumerate(zip(list(multi_swaps.keys())[:-1],
                                     list(multi_swaps.keys())[1:])):
            original = eig.copy()
            indices_to_swap = list(multi_swaps.values())[n]
            for k in indices_to_swap:
                index = np.argmin(np.abs(original[indices_to_swap, i+1] - original[k, i]))
                eig[k, i+1:j+1] = original[indices_to_swap[index], i+1:j+1]
                new_indices[k, i+1:j+1] = indices_to_swap[index]
                
        # deal with the last swap
        original = eig.copy()
        i = list(multi_swaps.keys())[-1]
        indices_to_swap = list(multi_swaps.values())[-1]
        for k in indices_to_swap:
            index = np.argmin(np.abs(original[indices_to_swap, i+1] - original[k, i]))
            eig[k, i+1] = original[indices_to_swap[index], i+1]
            new_indices[k, i+1] = indices_to_swap[index]
        
    return eig, new_indices
# %%
def calculate_sensitivity_dict(network):
    """ 
     This function first resorts the eigenvalues of the effective connectivity
    matrix such that their identity stays the same across frequencies.
    
    Then the frequency which is closest to complex(1,0) is identified per 
    eigenvalue trajectory.
    
    After evaluating the sensitivity measure, its projections on the 
    direction that influences the amplitude and the direction that influences
    the frequency are calculated.
    
    The results are stored in dictionary 
    """
    
    frequencies = network.analysis_params['omegas']/(2.*np.pi)
    eigenvalue_spectra = np.linalg.eig(
            nnmt.lif.exp.effective_connectivity(network))[0].T
    
    resorted_eigenvalue_spectra, new_indices = resort_eigenvalues(
        eigenvalue_spectra)
    
    # identify frequency which is closest to the point complex(1,0) per eigenvalue trajectory    
    sensitivity_dict = defaultdict(int)
    
    for eig_index, eig in enumerate(resorted_eigenvalue_spectra):
        critical_frequency = frequencies[np.argmin(abs(eig-1.0))]
        critical_frequency_index = np.argmin(abs(frequencies-critical_frequency))
        critical_eigenvalue = eig[critical_frequency_index]

        # calculate sensitivity measure at frequency
        omega = np.array([critical_frequency * 2 * np.pi])
        network_at_single_critical_frequency = network.change_parameters({
            'omegas': omega
            })
        
        nnmt.lif.exp.working_point(network_at_single_critical_frequency, method='taylor')
        nnmt.network_properties.delay_dist_matrix(network_at_single_critical_frequency)
        nnmt.lif.exp.transfer_function(network_at_single_critical_frequency, method='taylor')
        nnmt.lif.exp.effective_connectivity(network_at_single_critical_frequency)
        sensitivity_measure = nnmt.lif.exp.sensitivity_measure(network_at_single_critical_frequency)[0]
        
        # vector pointing from critical eigenvalue at frequency to complex(1,0)
        # perturbation shifting critical eigenvalue along k
        # brings eigenvalue towards or away from one, 
        # resulting in an increased or 
        # decreased peak amplitude in the spectrum
        k = np.asarray([1, 0])-np.asarray([critical_eigenvalue.real,
                                           critical_eigenvalue.imag])
        # normalize k
        k /= np.sqrt(np.dot(k, k))

        # vector perpendicular to k
        # perturbation shifting critical eigenvalue along k_per
        # alters the trajectory such that it passes closest 
        # to one at a lower or
        # higher frequency while conserving the height of the peak
        k_per = np.asarray([-k[1], k[0]])
        # normalize k_per
        k_per /= np.sqrt(np.dot(k_per, k_per))

        # projection of sensitivity measure in to direction that alters amplitude
        sensitivity_measure_amp = sensitivity_measure.real*k[0] + \
                                  sensitivity_measure.imag*k[1]
        # projection of sensitivity measure in to direction that alters frequency
        sensitivity_measure_freq = sensitivity_measure.real*k_per[0] + \
                                   sensitivity_measure.imag*k_per[1]

        sensitivity_dict[eig_index] = {
            'critical_frequency': critical_frequency,
            'critical_frequency_index': critical_frequency_index,
            'critical_eigenvalue': critical_eigenvalue,
            'k': k,
            'k_per': k_per,
            'sensitivity_measure': sensitivity_measure,
            'sensitivity_measure_amp': sensitivity_measure_amp,
            'sensitivity_measure_freq': sensitivity_measure_freq}
        
    return sensitivity_dict, resorted_eigenvalue_spectra


# %%
# create network model microcircuit
microcircuit = nnmt.models.Microcircuit(
    network_params='../tests/fixtures/integration/config/Bos2016_network_params.yaml',
    analysis_params='../tests/fixtures/integration/config/Bos2016_analysis_params.yaml')
# %%
frequencies = microcircuit.analysis_params['omegas']/(2.*np.pi)
# %%
# calculate working point for exponentially shape post synaptic currents
nnmt.lif.exp.working_point(microcircuit, method='taylor')
# calculate the transfer function
nnmt.lif.exp.transfer_function(microcircuit, method='taylor')
# calculate the delay distribution matrix
nnmt.network_properties.delay_dist_matrix(microcircuit)
# %%
# currently the sensitivity measure is evaluated for all frequencies that are
# defined via the analysis_params.
# Most often, a user will want to evaluate it only at particular frequencies.
# This requires now, to redefine the analysis frequencies, which is suboptimal
# as the other results for the former analysis frequencies will be overwritten.
# %% 

sensitivity_dict, resorted_eigenvalue_spectra = calculate_sensitivity_dict(microcircuit)
# %%
for i in range(8):
    print(sensitivity_dict[i]['critical_frequency'])
# %%
eigenvalues_to_be_plotted = [0, 1, 2, 3, 5, 6]
print(f'Eigenvalues to be plotted: {eigenvalues_to_be_plotted}')

# %%
power_spectra = nnmt.lif.exp.power_spectra(microcircuit).T


# %%
# Calculate the eigenvalue trajectories and sensitivity dict for isolated layer
# by redefining the connectivity matrix
microcircuit_isolated_layers = nnmt.models.Microcircuit(
    network_params='../tests/fixtures/integration/config/Bos2016_network_params.yaml',
    analysis_params='../tests/fixtures/integration/config/Bos2016_analysis_params.yaml')
# %%
full_indegree_matrix = microcircuit_isolated_layers.network_params['K']
frequencies = microcircuit_isolated_layers.analysis_params['omegas']/(2.*np.pi)
# %% 
# test for one layer
eigenvalue_spectra_isolated_layers = defaultdict(int)
power_spectra_isolated_layers = defaultdict(int)

i=0
reducing_matrix = np.zeros((8,8))
reducing_matrix[2*i:2*i+2, 2*i:2*i+2] = np.ones([2,2])
# for layer i, set indegree matrix such that it is isolated
microcircuit_isolated_layers.network_params.update(
    K = full_indegree_matrix*reducing_matrix)

# calculate working point for exponentially shape post synaptic currents
print(nnmt.lif.exp.working_point(microcircuit_isolated_layers, method='taylor'))
# calculate the transfer function
nnmt.lif.exp.transfer_function(microcircuit_isolated_layers, method='taylor')
# calculate the delay distribution matrix
nnmt.network_properties.delay_dist_matrix(microcircuit_isolated_layers)
eigenvalue_spectra_layer = np.linalg.eig(
        nnmt.lif.exp.effective_connectivity(microcircuit_isolated_layers))[0].T

power_spectra_layer = nnmt.lif.exp.power_spectra(microcircuit_isolated_layers).T


resorted_eigenvalue_spectra_layer, new_indices_layer = resort_eigenvalues(
        eigenvalue_spectra_layer)

eigenvalue_spectra_isolated_layers[i] = eigenvalue_spectra_layer   
power_spectra_isolated_layers[i] = power_spectra_layer  

# %%
freqs = microcircuit_isolated_layers.analysis_params['omegas']/(2*np.pi)
plt.semilogy(freqs, power_spectra_isolated_layers[i][0])

# %%
for i in range(8):
    plt.scatter(eigenvalue_spectra_isolated_layers[0].real,
                eigenvalue_spectra_isolated_layers[0].imag)
# %%

# restore the full indegree matrix again
microcircuit_isolated_layers.network_params.update(
    K = full_indegree_matrix)

# %% 
# Alter the indegree matrix to consist just of the individual isolated layers
isolated_layers_results = defaultdict(str)
for i, layer in enumerate(['23', '4', '5', '6']):
    isolated_layers_results[layer] = defaultdict(str)

    reducing_matrix = np.zeros((8,8))
    reducing_matrix[2*i:2*i+2, 2*i:2*i+2] = np.ones([2,2])
    # for layer i, set indegree matrix such that it is isolated
    # microcircuit_isolated_layers.network_params.update(
    #     K = full_indegree_matrix*reducing_matrix)
    isolated_layers_results[layer]['network'] = microcircuit.change_parameters({
        'K': full_indegree_matrix*reducing_matrix
    })
    
    # calculate working point for exponentially shape post synaptic currents
    nnmt.lif.exp.working_point(isolated_layers_results[layer]['network'], method='taylor')
    # calculate the transfer function
    nnmt.lif.exp.transfer_function(isolated_layers_results[layer]['network'], method='taylor')
    # calculate the delay distribution matrix
    nnmt.network_properties.delay_dist_matrix(isolated_layers_results[layer]['network'])
    eigenvalue_spectra_layer = np.linalg.eig(
            nnmt.lif.exp.effective_connectivity(isolated_layers_results[layer]['network']))[0].T
    # calculate the power spectra
    power_spectra_layer = nnmt.lif.exp.power_spectra(isolated_layers_results[layer]['network']).T
    
    resorted_eigenvalue_spectra_layer, new_indices_layer = resort_eigenvalues(
        eigenvalue_spectra_layer)
    
    isolated_layers_results[layer]['eigenvalue_spectra'] = eigenvalue_spectra_layer
    isolated_layers_results[layer]['power_spectra'] = power_spectra_layer  
# %%
for i, layer in enumerate(['23', '4', '5', '6']):
    layer = isolated_layers_results[layer]
    layer['sensitivity_dict'] = defaultdict(int)
        
    for eig_index, eig in enumerate(layer['eigenvalue_spectra']):
        critical_frequency = freqs_isolated_layers[np.argmin(abs(eig-1.0))]
        critical_frequency_index = np.argmin(abs(frequencies-critical_frequency))
        critical_eigenvalue = eig[critical_frequency_index]
    
        layer['sensitivity_dict'] [eig_index] = {
                'critical_frequency': critical_frequency,
                'critical_frequency_index': critical_frequency_index,
                'critical_eigenvalue': critical_eigenvalue}
# %%
for i, layer in enumerate(['23', '4', '5', '6']):
    for j in range(2):
        print(layer, j, isolated_layers_results[layer]['sensitivity_dict'][j])

layer_eigenvalue_tuples_to_be_plotted = [('23', 1), ('4', 1), ('5', 1), ('6', 1), ('23', 0), ('5', 0)]

# %%
for layer in range(4):
    for i in range(8):
        plt.scatter(eigenvalue_spectra_isolated_layers[layer].real,
                    eigenvalue_spectra_isolated_layers[layer].imag)
# %%
# %%
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
N = resorted_eigenvalue_spectra.shape[1]
dc = 1/float(N)
for i in range(0, N, 3):
    ax.plot(resorted_eigenvalue_spectra.T[i].real, 
            resorted_eigenvalue_spectra.T[i].imag, '.',
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
    ax.plot(resorted_eigenvalue_spectra.T[i].real, 
            resorted_eigenvalue_spectra.T[i].imag, '.',
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

# top
ax = fig.add_subplot(gsB[0])
N = resorted_eigenvalue_spectra.shape[1]
dc = 1/float(N)
for i, layer in enumerate(['23', '4', '5', '6']):
    # [:, 2*layer:2*layer+2] serves to plot only the non-zero eigenspectra
    plt.scatter(isolated_layers_results[layer]['eigenvalue_spectra'][:, :].real,
                isolated_layers_results[layer]['eigenvalue_spectra'][:, :].imag,
                    color=colors[i][0],
                    s=1.0, zorder=1)
ax.scatter(1,0, s=15, color='r')
ax.set_ylim([-4, 6.5])
ax.set_xlim([-11.5, 2])
ax.set_xticks([-9, -6, -3, 0])
ax.set_yticks([-3, 0, 3, 6])
ax.set_ylabel('Im($\lambda(\omega)$)')

# bottom
ax = fig.add_subplot(gsB[1])
for i, layer in enumerate(['23', '4', '5', '6']):
    plt.scatter(isolated_layers_results[layer]['eigenvalue_spectra'].real,
                isolated_layers_results[layer]['eigenvalue_spectra'].imag,
                    color=colors[i][0],
                    s=1.0, zorder=1)
        # ,
        #             color=(0.9-0.9*i*dc, 0.9-0.9*i*dc, 0.9-0.9*i*dc),
        #             markersize=1.0, zorder=1)
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
# box = ax.get_position()
# ax.set_position([box.x0, box.y0-box.height, box.width, box.height*2])
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
        
        
plt.savefig('Bos2016_Fig4.png')

# %%
