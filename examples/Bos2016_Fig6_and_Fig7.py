# %%
from nnmt.models import network
import nnmt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
import h5py_wrapper.wrapper as h5
from collections import defaultdict


plt.style.use('frontiers.mplstyle')
# %%

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
    
    print('Looping through eigenvalues...')
    
    for eig_index, eig in enumerate(resorted_eigenvalue_spectra):
        print(eig_index)
        critical_frequency = frequencies[np.argmin(abs(eig-1.0))]
        critical_frequency_index = np.argmin(abs(frequencies-critical_frequency))
        critical_eigenvalue = eig[critical_frequency_index]

        # calculate sensitivity measure at frequency
        omega = np.array([critical_frequency * 2 * np.pi])
        network_at_single_critical_frequency = network.change_parameters(
            changed_analysis_params={'omegas': omega})
            
        print(omega)
        
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
    print(sensitivity_dict[i]['critical_eigenvalue'])
    print(sensitivity_dict[i]['k'])
    print(sensitivity_dict[i]['k_per'])


    
    
eigenvalues_to_plot_high = [1, 0, 3, 2]
eigenvalue_to_plot_low = 6
# %%
from mpl_toolkits.axes_grid1 import make_axes_locatable

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

# %%
# two column figure, 180 mm wide
fig = plt.figure(figsize=(7.08661, 7.08661/2),
                 constrained_layout=True)
grid_specification = gridspec.GridSpec(2, 2, figure=fig)
colormap = 'coolwarm'
labels = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']




for ev, subpanel in zip(eigenvalues_to_plot_high, grid_specification):
    gs = gridspec.GridSpecFromSubplotSpec(1, 2, 
                                           width_ratios=[1, 1], 
                                           subplot_spec=subpanel)

    
    # senstivity_measure_amplitude
    ax = fig.add_subplot(gs[0])
    
    frequency = sensitivity_dict[ev]['critical_frequency']
    projection_of_sensitivity_measure = sensitivity_dict[ev]['sensitivity_measure_amp']
    
    # obtain maximal absolute value
    z = np.max(abs(projection_of_sensitivity_measure))
    rounded_frequency = str(np.round(frequency,1))

    plot_title = '$Z_{amp}$ @' + f'{rounded_frequency}'
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
    projection_of_sensitivity_measure = sensitivity_dict[ev]['sensitivity_measure_freq']
    
    # obtain maximal absolute value
    z = np.max(abs(projection_of_sensitivity_measure))
    rounded_frequency = str(np.round(frequency,1))

    plot_title = '$Z_{freq}$ @' + f'{rounded_frequency}'
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
    
    
plt.savefig('Bos2016_Fig6.png')

# %%
fix_path = '../tests/fixtures/integration/data/'
result = h5.load(fix_path + 'Bos2016_publicated_and_converted_data.h5')
# %%
result['sensitivity_measure'].keys()
result['sensitivity_measure']['eigs'].shape
result['sensitivity_measure']['freqs'].shape
result['sensitivity_measure']['high_gamma2']
# %%

# %%
# two column figure, 180 mm wide
fig = plt.figure(figsize=(3.34646, 3.34646/2),
                 constrained_layout=True)
colormap = 'coolwarm'
labels = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']

ax = fig.add_subplot(111)

ev = eigenvalue_to_plot_low

frequency = sensitivity_dict[ev]['critical_frequency']
projection_of_sensitivity_measure = sensitivity_dict[ev]['sensitivity_measure_amp']

# obtain maximal absolute value
z = np.max(abs(projection_of_sensitivity_measure))
rounded_frequency = str(np.round(frequency,1))

plot_title = '$Z_{amp}$ @' + f'{rounded_frequency}'
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

plt.savefig('Bos2016_Fig7.png')


# %%
