"""
Prediction and simulation of a simple E-I network
=================================================

Here we show how to use NNMT to predict the results of a simulation of a simple
E-I network with an excitatory (E) and an inhibitory (I) population. The
example has been adapted from the code published with :cite:t:`layer2023`,
where the NEST simulation has been based on the NEST example ...

The network properties are defined in ``config/network_params.yaml``.

First a network is created using the simulation platform NEST (add citation).
Then the connectivity matrix is extracted to be used with NNMT.  Then firing
rates, CVs, and pairwise correlations are computed. Subsequently, the
theoretical predictions for these quantities are computed using NNMT. Then the
we compute the correlation coefficient between prediction and simulation
results and plot the data.

Note that NEST uses non SI units as standards, while NNMT uses SI units.

Note that the simulated network is very small. It most likely contains
correlations that are not accounted for in mean-field theory.
"""


###############################################################################
# Imports
# -------

import string
import matplotlib.pyplot as plt
import nest
import random
import numpy as np
import nnmt

import utils
from utils import (colors, fontsize, labelsize, panel_label_fontsize)

config_path = 'config/'
simulation_data_path = 'data/'
config_path = 'config/'
theory_data_path = 'data/'
intermediate_results_path = 'data/'
plotting_data_path = 'data/'
plot_path = 'plots/'

network_param_file = 'config/network_params.yaml'
sim_param_file = 'config/simulation_params.yaml'
analysis_param_file = config_path + 'analysis_params.yaml'

network_id = 'ei_network'

sample_size = 10000

run = True

###############################################################################
# Helper functions
# ----------------
#
# The script starts with the definition of all the functions used. The main
# script starts below.

def adjust_weights(neurons, W):
    """
    Sets the connectivity matrix W, but same connections need to exist already.
    """
    syn_collection = nest.GetConnections(neurons)
    targets = np.array([t for t in syn_collection.targets()]) - 1
    sources = np.array([s for s in syn_collection.sources()]) - 1
    syn_collection.set({'weight': W[targets, sources]})


def connectivity(neurons):
    """ retrieves connectivity matrix from nest simulation """
    J = np.zeros((len(neurons), len(neurons)))
    syn_collection = nest.GetConnections(neurons)
    # get all targets, sources, and weights
    targets = np.array([t for t in syn_collection.targets()]) - 1
    sources = np.array([s for s in syn_collection.sources()]) - 1
    weights = np.array(syn_collection.get('weight'))

    # check for multapses
    # sort such that equal connections are next to each other
    idx = np.lexsort((sources, targets))
    targets = targets[idx]
    sources = sources[idx]
    weights = weights[idx]

    # calculate difference of subsequent elements and get position of zeros
    comb = np.vstack((targets, sources)).T
    mask = (np.diff(comb, axis=0) == [0, 0]).all(axis=1)
    mask = np.append(mask, [False])
    multapse_idx = np.where(mask)[0]

    # if multapses exist
    if len(multapse_idx) != 0:

        # split seperate connections
        split_idx = np.where(np.diff(multapse_idx) != 1)[0] + 1
        multapse_idx = np.split(multapse_idx, split_idx)

        # combine consecutive contributions: run over multapses and replace
        # multapse by respective single connection
        for m in multapse_idx:
            m = np.append(m, m[-1] + 1)
            weights[m[0]] = weights[m].sum()
            targets = np.delete(targets, m[1:])
            sources = np.delete(sources, m[1:])
            weights = np.delete(weights, m[1:])

    # get respective connectivity matrix
    J[targets, sources] = weights

    return J


def get_spike_trains(senders, times, id_min, id_max):
    """Returns a list of spike trains sorted by senders."""
    sorted_ids = np.lexsort([times, senders])
    senders = senders[sorted_ids]
    times = times[sorted_ids]
    spike_trains = np.split(times, np.where(np.diff(senders))[0]+1)
    missing_senders = (set(np.arange(id_min, id_max + 1))
                       - set(np.unique(senders)))
    for id in sorted(missing_senders):
        spike_trains = (
            spike_trains[:id-id_min]
            + [np.array([])] + spike_trains[id-id_min:])
    return spike_trains


def bin_spiketrains(spiketrains, t_min, t_max, bin_width):
    """
    Bins the spiketrains from `t_min` to `t_max` with bin size `bin_width`.
    """
    bins = np.arange(t_min, t_max + bin_width, bin_width)
    clipped_spiketrains = [st[st < t_max] for st in spiketrains]
    clipped_spiketrains = [st[st >= t_min] for st in clipped_spiketrains]
    binned_spiketrains = np.array(
        [np.histogram(st, bins)[0] for st in clipped_spiketrains]
    )
    return binned_spiketrains


def remove_and_shift_init_from_spiketrains(spiketrains, T_init=0):
    """
    Remove initialization time from spiketrains and shift them respectively.
    """
    return [st[st >= T_init] - T_init for st in spiketrains]


def remove_inactive_neurons(spiketrains):
    """
    Remove all neurons for which an CV could not be calculated.

    This are all neurons which fired no more than 2 spikes.
    """
    nspikes = np.array([len(st) for st in spiketrains])
    return [spiketrains[id] for id in np.where(nspikes >= 3)[0]]


def get_spike_count_covariances(binned_spiketrains):
    return np.cov(binned_spiketrains)


def spectral_bound(matrix):
    return np.linalg.eigvals(matrix).real.max()


def calc_rates(spiketrains, T, T_init=0):
    """
    Calculates single neuron rates from spiketrains

    Parameters
    ----------
    spiketrains : list
        list of recorded spiketimes
    T : float
        Total simulation time including initialization in ms
    T_init : float
        Initialization time in ms

    Results
    -------
    array
        Rates in Hz
    """
    T -= T_init
    clipped_spiketrains = [st[st >= T_init] for st in spiketrains]
    rates = np.array([len(st) / T for st in clipped_spiketrains])
    return rates


def calc_isis(spiketrains):
    """
    Calculates the inter-spike-intervals.
    """
    return [np.diff(st) for st in spiketrains]


def calc_cvs(spiketrains):
    """
    Calculate coefficients of variation of interspike intervals.
    """
    isis = calc_isis(spiketrains)
    stds = np.array([isi.std() for isi in isis])
    means = np.array([isi.mean() for isi in isis])
    return stds / means


def calc_autocorr(x):
    """Experimental implementation."""
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]


def scatter_plot(ax, data1, data2, s=0.5, diag=True, set_aspect=True,
                 **kwargs):
    ax.scatter(data1, data2, s=s, **kwargs)
    if diag:
        lower, upper = get_extrema([data1, data2])
        plot_diagonal(ax, lower, upper)
    if set_aspect:
        ax.set_aspect('equal', 'box')


def get_extrema(datasets):
    lower = min([data.min() for data in datasets])
    upper = max([data.max() for data in datasets])
    return lower, upper


def plot_diagonal(ax, lower, upper, color='black', zorder=-100, **kwargs):
    lower -= 0.1 * (upper - lower)
    upper += 0.1 * (upper - lower)
    diagonal = np.linspace(lower, upper)
    ax.plot(diagonal, diagonal, color=color, zorder=zorder, **kwargs)


def add_corrcoef(ax, cc, x=0.7, y=0.05):
    ax.text(x, y, f'$\\rho={cc:.2g}$', transform=ax.transAxes)


def plot_rates_scatter_plot(ax_rates, rates_sim, rates_thy, cc_rate, N_E):

    lower, upper = get_extrema([rates_sim, rates_thy])
    plot_diagonal(ax_rates, lower, upper, color='lightgray')
    scatter_plot(ax_rates, rates_sim[:N_E], rates_thy[:N_E], color=blue,
                 label='E', diag=False, rasterized=True)
    scatter_plot(ax_rates, rates_sim[N_E:], rates_thy[N_E:], color=red,
                 label='I', diag=False, rasterized=True)
    add_corrcoef(ax_rates, cc_rate)


def plot_corrs_scatter_plot(ax_corrs,
                            corrs_sim_EE, corrs_sim_EI, corrs_sim_II,
                            corrs_thy_EE, corrs_thy_EI, corrs_thy_II,
                            cc_corr):

    lower, upper = get_extrema([corrs_sim_EE, corrs_thy_EE,
                                corrs_sim_EI, corrs_thy_EI,
                                corrs_sim_II, corrs_thy_II])
    plot_diagonal(ax_corrs, lower, upper, color='lightgray')
    scatter_plot(ax_corrs, corrs_sim_EE, corrs_thy_EE, diag=False,
                 color=blue, alpha=alpha, label='EE', rasterized=True, zorder=3)
    scatter_plot(ax_corrs, corrs_sim_EI, corrs_thy_EI, diag=False,
                 color=yellow, alpha=alpha, label='EI', rasterized=True, zorder=2)
    scatter_plot(ax_corrs, corrs_sim_II, corrs_thy_II, diag=False,
                 color=red, alpha=alpha, label='II', rasterized=True, zorder=1)
    add_corrcoef(ax_corrs, cc_corr)


def raster_plot(ax, spiketrains, samples=[20, 5], t_min=1, t_max=5,
                colors=['steelblue', 'red'], marker='ticks', **kwargs):
    slices = np.append([0], np.array(samples)).cumsum()
    for i in range(len(samples)):
        sts = random.choices(spiketrains[i], k=samples[i])
        sts = [st[(st > t_min) & (st < t_max)] for st in sts]
        neuron_ids = np.arange(slices[i], slices[i+1])
        for id, spiketrain in zip(neuron_ids + 1, sts):
            if marker == 'ticks':
                ax.scatter(spiketrain, id * np.ones(len(spiketrain)),
                        color=colors[i], marker=2, s=8, **kwargs)
            elif marker == 'dots':
                ax.scatter(spiketrain, id * np.ones(len(spiketrain)),
                        color=colors[i], s=0.5, **kwargs)
            else:
                raise RuntimeError(f'Marker {marker} unknown.')
    ax.set_ylim([slices[-1]+1, -1])


if run:

    ###############################################################################
    # Drawing a connectivity matrix with NEST
    # ---------------------------------------
    #
    # First, we need to draw a random connectivity matrix, for which we use NEST.
    #
    # We start by loading the parameters from the yaml files in which we used the
    # units that NEST is expecting. Here we load the parameters in these units. For
    # example, time constants like ``tau_m`` are given in ms.

    network_params = nnmt.input_output.load_val_unit_dict_from_yaml(
        network_param_file)
    nnmt.utils._strip_units(network_params)

    sim_params = nnmt.input_output.load_val_unit_dict_from_yaml(
        sim_param_file)
    nnmt.utils._strip_units(sim_params)


    ###############################################################################
    # Here we define the parameters for building the network.

    connection_rule = network_params['connection_rule']
    multapses = network_params['multapses']
    neuron_type = network_params['neuron_type']
    gaussianize = network_params['gaussianize']

    N_E = network_params['N'][0]  # number of excitatory neurons
    N_I = network_params['N'][1]  # number of inhibitory neurons
    N = network_params['N'].sum()  # number of neurons in total

    p = network_params['p']  # connection probability
    K_E = int(p * N_E)  # number of excitatory synapses per neuron
    K_I = int(p * N_I)  # number of inhibitory synapses per neuron

    tau_m = network_params['tau_m']  # time const of membrane potential in ms
    tau_r = network_params['tau_r']  # refactory time in ms
    tau_s = network_params['tau_s']  # synaptic time constant in ms
    delay = network_params['d']  # synaptic delay in ms

    V_r = network_params['V_0_rel']  # reset potential in mV
    V_th = network_params['V_th_rel']  # membrane threshold potential in mV
    V_m = network_params['V_m']  # initial potential in mV
    E_L = network_params['E_L']  # resting potential in mV

    g = network_params['g']  # absolute ratio inhibitory weight/excitatory weight
    J = network_params['j']  # postsynaptic amplitude in mV
    J_ex = J  # amplitude of excitatory postsynaptic current in mV
    J_in = -g * J_ex  # amplitude of inhibitory postsynaptic current in mV

    p_rate_E = network_params['nu_ext'][0]  # external excitatory noise rate in Hz
    p_rate_I = network_params['nu_ext'][1]  # external inhibitory noise rate in Hz
    I_ext = network_params['I_ext']  # external DC current in pA

    C = network_params['C']  # membrane capacitance in pF


    ###############################################################################
    # The parameters of the neurons are stored in a dictionary.

    neuron_params = {
        "C_m": C,
        "tau_m": tau_m,
        "t_ref": tau_r,
        "E_L": E_L,
        "V_reset": V_r,
        "V_m": V_m,
        "V_th": V_th,
        "I_e": I_ext
        }

    if neuron_type == 'iaf_psc_exp':
        neuron_params['tau_syn_ex'] = tau_s
        neuron_params['tau_syn_in'] = tau_s


    ###############################################################################
    # Configuration of the NEST simulation kernel.
    #
    # As we want to simulate the network later, we need to set these properties
    # here. They cannot be changed, after NEST nodes have been created.
    nest.ResetKernel()
    nest.local_num_threads = sim_params['local_num_threads']
    nest.resolution = sim_params['dt']


    ###############################################################################
    # Creation of the nodes and external noise generators

    print("Building network")

    if neuron_type == 'iaf_psc_delta':
        nodes_ex = nest.Create("iaf_psc_delta", N_E, params=neuron_params)
        nodes_in = nest.Create("iaf_psc_delta", N_I, params=neuron_params)
    elif neuron_type == 'iaf_psc_exp':
        nodes_ex = nest.Create("iaf_psc_exp", N_E, params=neuron_params)
        nodes_in = nest.Create("iaf_psc_exp", N_I, params=neuron_params)
    neurons = nodes_ex + nodes_in


    ###############################################################################
    # Definition of a synapses

    nest.CopyModel("static_synapse", "excitatory",
                    {"weight": J_ex, "delay": delay})
    nest.CopyModel("static_synapse", "inhibitory",
                    {"weight": J_in, "delay": delay})

    syn_params_ex = {"synapse_model": "excitatory"}
    syn_params_in = {"synapse_model": "inhibitory"}


    ###############################################################################
    # Set seeds; put seeding here, because for some reason seed is reset if
    # defined further above

    np.random.seed(sim_params['np_seed'])
    nest.rng_seed = sim_params['seed']


    ###############################################################################
    # Connecting the populations

    print("Connecting network")

    if connection_rule == 'pairwise_bernoulli':
        conn_params_ex = {'rule': 'pairwise_bernoulli', 'p': p,
                        'allow_multapses': multapses}
        conn_params_in = {'rule': 'pairwise_bernoulli', 'p': p,
                        'allow_multapses': multapses}
    elif connection_rule == 'fixed_indegree':
        conn_params_ex = {'rule': 'fixed_indegree', 'indegree': K_E,
                        'allow_multapses': multapses}
        conn_params_in = {'rule': 'fixed_indegree', 'indegree': K_I,
                        'allow_multapses': multapses}
    else:
        raise ValueError(f'Connection rule {connection_rule} not implemented!')

    print("Excitatory connections")
    nest.Connect(nodes_ex, nodes_ex + nodes_in, conn_params_ex, syn_params_ex)

    print("Inhibitory connections")
    nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_in, syn_params_in)


    ###############################################################################
    # Extract connectivity matrix

    print("Extracting connectivity matrix")
    W = connectivity(neurons)


    ###############################################################################
    # Gaussianize connectivity matrix

    if gaussianize:
        print("Gaussianizing connectivity matrix")
        J_std = network_params['j_std'] * np.abs(J_ex)

        W[W != 0] += np.random.normal(
            loc=0, scale=J_std, size=np.count_nonzero(W))

        adjust_weights(neurons, W)



    ###############################################################################
    # Prediction using NNMT
    # ---------------------
    #
    # Next, we want to predict the single-neuron resolved rates, CVs and pairwise
    # correlations given the connectivity matrix.
    #
    # We start by building a NNMT network. Therefore, we load the network
    # parameters again. But this time we convert the units directly to SI.

    # Load yaml file to SI units
    network = nnmt.models.Plain(network_param_file)

    # Add newly calculated parameters to network parameter dictionary
    new_network_params = dict(
        K=np.where(W != 0, 1, 0),  # here we assume that multapses are not allowed!
        J=W/1000,  # convert from mV to V
        K_ext=np.vstack([np.ones(N), np.ones(N)]).T,
        J_ext=np.vstack([np.ones(N) * J_ex, np.ones(N) * J_in]).T / 1000  # mV to V
    )
    network.network_params.update(new_network_params)

    # only needed for population values
    network_params['K_E'] = K_E
    network_params['K_I'] = K_I

    ###########################################################################

    print('Build NNMT model')
    network_file = (
        theory_data_path + network_id + '.h5')

    if neuron_type == 'iaf_psc_delta':
        network.network_params['tau_s'] = 0.0

    print('Estimate firing rates')
    working_point = nnmt.lif.exp.working_point(network)
    rates_thy = working_point['firing_rates']

    print('Estimate CVs')
    cvs_thy = nnmt.lif.exp.cvs(network)

    print('Estimate effective connectivity')
    W_eff = nnmt.lif.exp.pairwise_effective_connectivity(network)

    print('Estimate spectral bound')
    r = nnmt.lif.exp.spectral_bound(network)

    print('Estimate pairwise covarianes and correlations')
    covs_thy = nnmt.lif.exp._pairwise_covariances(W_eff, rates_thy, cvs_thy)
    std_thy = np.sqrt(np.diag(covs_thy))
    corrs_thy = covs_thy / np.outer(std_thy, std_thy)

    print(f'rates_thy: {rates_thy.mean()} +- {rates_thy.std()}')
    print(f'cvs_thy: {cvs_thy.mean()} +- {cvs_thy.std()}')
    print(f'r_thy: {r}')


    # ###########################################################################
    # Population mean-field theory
    print('Build NNMT population model')

    K_pop = np.array([[K_E, K_I], [K_E, K_I]])
    J_pop = np.array([[J_ex, J_in],
                      [J_ex, J_in]])
    # convert from mV to V
    J_pop /= 1000

    K_ext_pop = np.array([[1, 1],
                          [1, 1]])
    J_ext_pop = np.array([[J_ex, J_in],
                          [J_ex, J_in]])
    # convert from mV to V
    J_ext_pop /= 1000

    nu_ext_pop = network_params['nu_ext']

    pop_params = {
        'K': K_pop,
        'J': J_pop,
        'tau_m': tau_m,
        'tau_s': tau_s,
        'tau_r': tau_r,
        'V_th_rel': V_th,
        'V_0_rel': V_r,
        'J_ext': J_ext_pop,
        'K_ext': K_ext_pop,
        'nu_ext': nu_ext_pop,
        'I_ext': I_ext,
        'C': C
    }

    pop_network = nnmt.models.Plain(network_param_file)
    pop_network.network_params['K'] = K_pop
    pop_network.network_params['J'] = J_pop
    pop_network.network_params['K_ext'] = K_ext_pop
    pop_network.network_params['J_ext'] = J_ext_pop

    print('Estimate population rates')
    nnmt.lif.exp.working_point(pop_network)
    rates_pop = nnmt.lif.exp.firing_rates(pop_network)

    print('Estimate population CVs')
    cvs_pop = nnmt.lif.exp.cvs(pop_network)


    ###############################################################################
    # Simulating the network
    # ----------------------

    print('Finish building NEST network.')


    ###############################################################################

    spike_recorder = nest.Create("spike_recorder")

    noise_E = nest.Create("poisson_generator", params={"rate": p_rate_E})
    noise_I = nest.Create("poisson_generator", params={"rate": p_rate_I})


    ###############################################################################
    # Connect devices

    print("Connecting devices")
    nest.Connect(nodes_ex + nodes_in, spike_recorder,
                syn_spec="excitatory")


    ###############################################################################
    # Connecting the previously defined poisson generators

    nest.Connect(noise_E, nodes_ex + nodes_in, syn_spec=syn_params_ex)
    nest.Connect(noise_I, nodes_ex + nodes_in, syn_spec=syn_params_in)


    ###############################################################################
    # Simulation of the network

    print("Simulating")

    simtime = sim_params['simtime']  # simulation time in ms

    nest.Simulate(simtime)


    ###############################################################################
    # Extract data

    print('Extract spiketrains')
    spiketrains = []
    events = spike_recorder.get('events')
    times = events['times']
    senders = events['senders']
    id_min = 1
    id_max = N
    spiketrains = get_spike_trains(senders, times, id_min, id_max)


    ###############################################################################
    # Analyzing the simulation results
    # --------------------------------
    #
    # Here we compute the firing rates, CVs, and pairwise correlations from the
    # simulation data.

    print('Analyze simulation results')

    ###############################################################################
    # load analysis params

    analysis_params = nnmt.input_output.load_val_unit_dict_from_yaml(
        analysis_param_file)
    nnmt.utils._convert_to_si_and_strip_units(analysis_params)

    T_init = analysis_params['T_init']
    binwidth = analysis_params['binwidth']


    ###############################################################################
    # analyze simulated data

    # convert units to SI units (from ms to s)
    simtime = sim_params['simtime'] / 1000
    spiketrains = [st / 1000 for st in spiketrains]

    # remove initialization time
    clipped_spiketrains = remove_and_shift_init_from_spiketrains(spiketrains,
                                                                    T_init)

    # calculate rates
    print('Calculate simulated rates')
    rates_sim = calc_rates(spiketrains, simtime, T_init)

    # calculate covs
    print('Calculate simulated CVs')
    cvs_sim = calc_cvs(clipped_spiketrains)

    del clipped_spiketrains

    print('Calculate simulated covariances and correlations')
    binned_spiketrains = bin_spiketrains(
        spiketrains, T_init, simtime, binwidth)
    covs_sim = np.cov(binned_spiketrains) / binwidth
    corrs_sim = np.corrcoef(binned_spiketrains)

    # del binned_spiketrains

    upper_triangle_indices = np.triu_indices_from(covs_sim, k=1)
    cross_covs_sim = covs_sim[upper_triangle_indices]
    cross_corrs_sim = corrs_sim[upper_triangle_indices]
    cross_covs_thy = covs_thy[upper_triangle_indices]
    cross_corrs_thy = corrs_thy[upper_triangle_indices]


    ###############################################################################
    # Saving of results

    results = dict(
        spiketrains=spiketrains,
        rates_thy=rates_thy,
        cvs_thy=cvs_thy,
        corrs_thy=corrs_thy,
        rates_sim=rates_sim,
        cvs_sim=cvs_sim,
        corrs_sim=corrs_sim,
        rates_pop=rates_pop,
        cvs_pop=cvs_pop,
    )

    np.savez(f'temp/{network_id}.npz',
            results=results,
            network_params=network_params,
            sim_params=sim_params,
            analysis_params=analysis_params,
            allow_pickle=True)


###############################################################################
# loading of results

print('Load data')
input_dict = np.load(f'temp/{network_id}.npz', allow_pickle=True)
network_params = input_dict['network_params'].tolist()
N_E = network_params['N'][0]

results = input_dict['results'].tolist()

rates_thy = results['rates_thy']
cvs_thy = results['cvs_thy']
corrs_thy = results['corrs_thy']
rates_sim = results['rates_sim']
cvs_sim = results['cvs_sim']
corrs_sim = results['corrs_sim']
spiketrains = results['spiketrains']
rates_pop = results['rates_pop']
cvs_pop = results['cvs_pop']


###############################################################################
# Separate results into differnet populations
# -------------------------------------------

print('Separate population results')

def sample_corrs(data, sample_ixs):
    return data[(sample_ixs[0])], data[(sample_ixs[1])], data[(sample_ixs[2])]

random.seed(42)

# Separate spiketrains

spiketrains_E = random.choices(spiketrains[:N_E], k=75)
spiketrains_I = random.choices(spiketrains[N_E:], k=25)

# Separate and subsample correlations

ix = np.triu_indices_from(corrs_thy, k=1)

cross_corrs_sim = corrs_sim[ix]
cross_corrs_thy = corrs_thy[ix]

ix = np.array(ix)
ix_EE = ix[:, (ix[0] < N_E) & (ix[1] < N_E)]
ix_EI = ix[:, (ix[0] < N_E) & (ix[1] >= N_E)]
ix_II = ix[:, (ix[0] >= N_E) & (ix[1] >= N_E)]
ixs = [ix_EE, ix_EI, ix_II]

np.random.seed(42)
sample_ixs = []
for ix in ixs:
    sample_ix = np.random.choice(
        np.arange(len(ix[0])), size=sample_size, replace=False)
    sample_ixs.append((ix[0][sample_ix], ix[1][sample_ix]))

corrs_thy_EE, corrs_thy_EI, corrs_thy_II = sample_corrs(corrs_thy, sample_ixs)
corrs_sim_EE, corrs_sim_EI, corrs_sim_II = sample_corrs(corrs_sim, sample_ixs)

sampled_cross_corrs_thy = np.append(corrs_thy_EE, [corrs_thy_EI, corrs_thy_II])
sampled_cross_corrs_sim = np.append(corrs_sim_EE, [corrs_sim_EI, corrs_sim_II])

###############################################################################
# Plotting of results
# -------------------
#
# Here we plot the firing rates, CVs, and pairwise covariances


###############################################################################
# options

print('Start plotting')
blue = colors['blue']
red = colors['red']
yellow = colors['yellow']
neutral = colors['lightneutral']
darkneutral = colors['neutral']

alpha = 1
markerscale = 5

width = utils.mm2inch(180)
height = utils.mm2inch(180)

fig = plt.figure(figsize=[width, height])

plt.rc('font', size=fontsize)
plt.rc('xtick', labelsize=labelsize)
plt.rc('ytick', labelsize=labelsize)

grid = (3, 3)

cc_rates = np.corrcoef(rates_sim, rates_thy)[0, 1]
cc_cvs = np.corrcoef(cvs_sim, cvs_thy)[0, 1]
cc_corrs = np.corrcoef(cross_corrs_sim, cross_corrs_thy)[0, 1]

ax_spiketrains = plt.subplot2grid(grid, (0, 0), colspan=3)
ax_rates_hist = plt.subplot2grid(grid, (1, 0))
ax_cvs_hist = plt.subplot2grid(grid, (1, 1))
ax_corrs_hist = plt.subplot2grid(grid, (1, 2))
ax_rates = plt.subplot2grid(grid, (2, 0))
ax_cvs = plt.subplot2grid(grid, (2, 1))
ax_corrs = plt.subplot2grid(grid, (2, 2))
axs = [ax_spiketrains,
       ax_rates_hist, ax_cvs_hist, ax_corrs_hist,
       ax_rates, ax_cvs, ax_corrs]

print('Plot spiketrains')
raster_plot(ax_spiketrains,
            [spiketrains_E, spiketrains_I],
            colors=[blue, red])
ax_spiketrains.set_title('Spiketrains', fontweight='bold')
ax_spiketrains.set_xlabel('time')

print('Plot histograms')

hist1 = ax_rates_hist.hist(rates_thy, bins=30, alpha=0.5, density=True,
                   label='thy',
                   color=neutral)
hist2 = ax_rates_hist.hist(rates_sim, bins=30, alpha=0.5, density=True,
                   label='sim',
                   color=darkneutral)
ax_rates_hist.set_title('Rates', fontweight='bold')
ax_rates_hist.set_ylim([0,0.21])
ax_rates_hist.set_xlabel('rate [Hz]')
ax_rates_hist.set_ylabel('pdf')
line = ax_rates_hist.axvline(rates_pop.mean(),
                      label='pop thy',
                      color='dimgrey',
                      linestyle='--')
#get handles and labels
handles, labels = ax_rates_hist.get_legend_handles_labels()

#specify order of items in legend
order = [1, 2, 0]

#add legend to plot
ax_rates_hist.legend(
    [handles[idx] for idx in order],[labels[idx] for idx in order],
    ncol=2, framealpha=1, bbox_to_anchor=(1, 1.04), loc='upper right')

ax_cvs_hist.hist(cvs_thy, bins=30, alpha=0.5, density=True, color=neutral)
ax_cvs_hist.hist(cvs_sim, bins=30, alpha=0.5, density=True, color=darkneutral)
ax_cvs_hist.set_title('CVs', fontweight='bold')
ax_cvs_hist.set_xlabel('CVs')
ax_cvs_hist.set_ylabel('pdf')
ax_cvs_hist.axvline(cvs_pop.mean(), color='dimgrey', linestyle='--')

ax_corrs_hist.hist(sampled_cross_corrs_thy, bins=30, alpha=0.5, density=True,
                   color=neutral)
ax_corrs_hist.hist(sampled_cross_corrs_sim, bins=30, alpha=0.5, density=True,
                   color=darkneutral)
ax_corrs_hist.set_title('Correlations', fontweight='bold')
ax_corrs_hist.set_ylabel('pdf')

print('Plot scatter plots')
plot_rates_scatter_plot(ax_rates, rates_sim, rates_thy, cc_rates, N_E)
ax_rates.set_xlabel('simulation')
ax_rates.set_ylabel('theory')
plot_rates_scatter_plot(ax_cvs, cvs_sim, cvs_thy, cc_cvs, N_E)
ax_cvs.set_xlabel('simulation')
ax_cvs.set_ylabel('theory')
plot_corrs_scatter_plot(ax_corrs,
                        corrs_sim_EE, corrs_sim_EI, corrs_sim_II,
                        corrs_thy_EE, corrs_thy_EI, corrs_thy_II,
                        cc_corrs)
ax_corrs.set_xlabel('simulation')
ax_corrs.set_ylabel('theory')

ax_rates.legend(markerscale=markerscale)
leg = ax_corrs.legend(markerscale=markerscale)
for lh in leg.legendHandles:
    lh.set_alpha(1)


labels = list(string.ascii_lowercase[:len(axs)])

x_positions = [-0.1] * len(axs)
x_positions[0] = -0.03  # Adjust the value as needed

utils.add_panel_labels(axs, labels, x_positions=x_positions,
                       fontsize=panel_label_fontsize,
                       use_parenthesis=True)
utils.remove_borders(axs)

plt.tight_layout()
plt.savefig(plot_path + network_id + '_thy_vs_sim.pdf', dpi=600)
plt.close()
