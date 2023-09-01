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
"""


###############################################################################
# Imports
# -------


import string
import matplotlib.pyplot as plt
import sys
import time
import nest
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


def cosine_similarity(a, b):
    return a.T @ b / np.sqrt(a.T @ a) / np.sqrt(b.T @ b)


def add_corrcoef(ax, cc, x=0.7, y=0.05):
    ax.text(x, y, f'$\\rho={cc:.2g}$', transform=ax.transAxes)


def add_cosine_similarity(ax, cs, x=0.7, y=0.12):
    ax.text(x, y, f'$cs={cs:.2g}$', transform=ax.transAxes)


def add_nrmse(ax, nrmse, x=0.7, y=0.19):
    ax.text(x, y, f'$nrmse={nrmse:.2g}$', transform=ax.transAxes)


def nrmse(a, b):
    rmse = np.sqrt(((a - b)**2).sum() / len(a))
    std = a.std()
    return rmse / std


def plot_row(axs,
             rates_sim, rates_thy,
             covs_sim_EE, covs_sim_EI, covs_sim_II,
             covs_thy_EE, covs_thy_EI, covs_thy_II,
             corrs_sim_EE, corrs_sim_EI, corrs_sim_II,
             corrs_thy_EE, corrs_thy_EI, corrs_thy_II,
             cc_rate, cc_cov, cc_corr,
             N_E
             ):

    plot_rates_scatter_plot(axs[0], rates_sim, rates_thy, cc_rate, N_E)
    plot_covs_scatter_plot(axs[1],
                           covs_sim_EE, covs_sim_EI, covs_sim_II,
                           covs_thy_EE, covs_thy_EI, covs_thy_II,
                           cc_cov)
    plot_corrs_scatter_plot(axs[2],
                            corrs_sim_EE, corrs_sim_EI, corrs_sim_II,
                            corrs_thy_EE, corrs_thy_EI, corrs_thy_II,
                            cc_corr)


def plot_rates_scatter_plot(ax_rates, rates_sim, rates_thy, cc_rate, N_E):

    lower, upper = get_extrema([rates_sim, rates_thy])
    plot_diagonal(ax_rates, lower, upper, color='lightgray')
    scatter_plot(ax_rates, rates_sim[:N_E], rates_thy[:N_E], color=blue,
                 label='E', diag=False, rasterized=True)
    scatter_plot(ax_rates, rates_sim[N_E:], rates_thy[N_E:], color=red,
                 label='I', diag=False, rasterized=True)
    add_corrcoef(ax_rates, cc_rate)


def plot_covs_scatter_plot(ax_covs,
                           covs_sim_EE, covs_sim_EI, covs_sim_II,
                           covs_thy_EE, covs_thy_EI, covs_thy_II,
                           cc_cov,
                           ):

    lower, upper = get_extrema([covs_sim_EE, covs_thy_EE,
                                covs_sim_EI, covs_thy_EI,
                                covs_sim_II, covs_thy_II])
    plot_diagonal(ax_covs, lower, upper, color='lightgray')
    scatter_plot(ax_covs, covs_sim_EE, covs_thy_EE, diag=False,
                 color=blue, alpha=alpha, label='EE', rasterized=True)
    scatter_plot(ax_covs, covs_sim_EI, covs_thy_EI, diag=False,
                 color=yellow, alpha=alpha, label='EI', rasterized=True)
    scatter_plot(ax_covs, covs_sim_II, covs_thy_II, diag=False,
                 color=red, alpha=alpha, label='II', rasterized=True)
    add_corrcoef(ax_covs, cc_cov)


def plot_corrs_scatter_plot(ax_corrs,
                            corrs_sim_EE, corrs_sim_EI, corrs_sim_II,
                            corrs_thy_EE, corrs_thy_EI, corrs_thy_II,
                            cc_corr):

    lower, upper = get_extrema([corrs_sim_EE, corrs_thy_EE,
                                corrs_sim_EI, corrs_thy_EI,
                                corrs_sim_II, corrs_thy_II])
    # lower, upper = -1, 1
    plot_diagonal(ax_corrs, lower, upper, color='lightgray')
    scatter_plot(ax_corrs, corrs_sim_EE, corrs_thy_EE, diag=False,
                 color=blue, alpha=alpha, label='EE', rasterized=True)
    scatter_plot(ax_corrs, corrs_sim_EI, corrs_thy_EI, diag=False,
                 color=yellow, alpha=alpha, label='EI', rasterized=True)
    scatter_plot(ax_corrs, corrs_sim_II, corrs_thy_II, diag=False,
                 color=red, alpha=alpha, label='II', rasterized=True)
    add_corrcoef(ax_corrs, cc_corr)


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
# network_params = nnmt.input_output.load_val_unit_dict_from_yaml(
#     network_param_file)
# nnmt.utils._convert_to_si_and_strip_units(network_params)

# Add newly calculated parameters to network parameter dictionary
new_network_params = dict(
    K=np.where(W != 0, 1, 0),  # here we assume that multapses are not allowed!
    J=W/1000,  # convert from mV to V
    K_ext=np.vstack([np.ones(N), np.ones(N)]).T,
    J_ext=np.vstack([np.ones(N) * J_ex, np.ones(N) * J_in]).T / 1000  # mV to V
)
network.network_params.update(new_network_params)

# network_param_dict = dict(
#     tau_m=tau_m,
#     tau_r=tau_r,
#     C=C,
#     V_th=V_th,
#     V_r=V_r,
#     J_ex=J_ex,
#     J_in=J_in,
#     delay=delay,
#     p=p,
#     K_E=K_E,
#     K_I=K_I,
#     N_E=N_E,
#     N_I=N_I,
#     K_ext=np.vstack([np.ones(N), np.ones(N)]).T,
#     J_ext=np.vstack([np.ones(N) * J_ex, np.ones(N) * J_in]).T,
# )

# if neuron_type == 'iaf_psc_delta':
#     network_param_dict['tau_s'] = 0.0
# elif neuron_type == 'iaf_psc_exp':
#     network_param_dict['tau_s'] = tau_s


# network_param_dict['W'] = W

# only needed for population values
network_params['K_E'] = K_E
network_params['K_I'] = K_I

###########################################################################

# # load parameters from simulation
# print('Load parameters from simulation')
# # load yaml file
# network_params = nnmt.input_output.load_val_unit_dict_from_yaml(
#     network_param_file)
# nnmt.utils._convert_to_si_and_strip_units(network_params)

# simulation_params = nnmt.input_output.load_val_unit_dict_from_yaml(
#     sim_param_file)
# nnmt.utils._convert_to_si_and_strip_units(simulation_params)

# N = network_params['N'].sum()
# p = network_params['p']
# j = network_params['j'] * 1000
# connection_rule = network_params['connection_rule']
# T = simulation_params['simtime'] * 1000

# # # load and integrate parameters saved in simulation
# # input_dict = np.load(params_from_sim, allow_pickle=True)
# # temp = network_params
# # network_params = input_dict['network_params'].tolist()
# # network_params.update(temp)

# network_params.pop('V_th')
# network_params.pop('V_r')
# network_params.pop('delay')
# network_params.pop('J_ex')
# network_params.pop('J_in')
# # convert mV used by NEST simulation to V used by NNMT
# network_params['W'] /= 1000
# network_params['J_ext'] /= 1000
# W = network_params['W']
# N_E = network_params['N_E']
# N_I = network_params['N_I']
# K_E = network_params['K_E']
# K_I = network_params['K_I']
# tau_r = network_params['tau_r']
# V_th = network_params['V_th_rel']
# V_r = network_params['V_0_rel']
# nu_ext = network_params['nu_ext']
# J_ex = network_params['j']
# J_in = - network_params['j'] * network_params['g']
# tau_m = network_params['tau_m']
# neuron_type = network_params['neuron_type']
# if neuron_type == 'iaf_psc_delta':
#     _prefix = 'lif.delta.'
#     tau_s = 0.0
# elif neuron_type == 'iaf_psc_exp':
#     _prefix = 'lif.exp.'
#     tau_s = network_params['tau_s']
# else:
#     raise RuntimeError(f'Unkown neuron type: {neuron_type}')
# C = network_params['C']
# K_ext = network_params['K_ext']
# # convert mV used by NEST simulation to V used by NNMT
# J_ext = network_params['J_ext']
# I_ext = network_params['I_ext']
# # assume no multapses allowed!
# K = np.where(W != 0, 1, 0)

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
# print('Saving')

# results_dict = dict(
#     rates_thy=rates_thy,
#     corrs_thy=corrs_thy,
#     covs_thy=covs_thy,
#     cvs_thy=cvs_thy,
#     r=r,
# )

# np.savez(
#     intermediate_results_path + f'{network_id}_thy.npz',
#     network_params=network_params,
#     results=results_dict)


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
# Save results

# print('Saving')
# results_dict = dict(
#     spiketrains=spiketrains,
# )

# np.savez(
#     simulation_data_path + '_spiketrains.npz',
#     results=results_dict)


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

# ###############################################################################
# # load network params

# network_params = nnmt.input_output.load_val_unit_dict_from_yaml(
#     network_param_file)
# nnmt.utils._convert_to_si_and_strip_units(network_params)

# ###############################################################################
# # load simulation params

# simulation_params = nnmt.input_output.load_val_unit_dict_from_yaml(
#     sim_param_file)
# nnmt.utils._convert_to_si_and_strip_units(simulation_params)

# ###############################################################################
# # load data

# N = network_params['N'].sum()
# p = network_params['p']
# j = network_params['j'] * 1000
# connection_rule = network_params['connection_rule']
# T = simulation_params['simtime'] * 1000

# input_dict = np.load(
#     simulation_data_path + network_id + '_spiketrains.npz',
#     allow_pickle=True)
# results = input_dict['results'].tolist()

# ###############################################################################
# # update params with parameters obtained through simulation

# input_dict = np.load(simulation_data_path + network_id + '_params.npz',
#                         allow_pickle=True)

# temp = network_params
# network_params = input_dict['network_params'].tolist()
# network_params.update(temp)

# temp = simulation_params
# simulation_params = input_dict['simulation_params'].tolist()
# simulation_params.update(temp)

# ###############################################################################
# # load parameters

# network_params.pop('V_th')
# network_params.pop('V_r')
# network_params.pop('delay')
# network_params.pop('J_ex')
# network_params.pop('J_in')
# network_params['W'] /= 1000
# network_params['J_ext'] /= 1000

# W = network_params['W']
# N_E = network_params['N_E']
# N_I = network_params['N_I']
# tau_r = network_params['tau_r']
# V_th = network_params['V_th_rel']
# V_r = network_params['V_0_rel']
# nu_ext = network_params['nu_ext']
# tau_m = network_params['tau_m']

# neuron_type = network_params['neuron_type']
# if neuron_type == 'iaf_psc_delta':
#     tau_s = 0.0
# else:
#     tau_s = network_params['tau_s']

# C = network_params['C']
# K_ext = network_params['K_ext']
# J_ext = network_params['J_ext']
# I_ext = network_params['I_ext']

# # assume no multapses allowed!
# K = np.where(W != 0, 1, 0)

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
# Separate results into differnet populations
# -------------------------------------------

ix = np.array(upper_triangle_indices)

ix_EE = ix[:, (ix[0] < N_E) & (ix[1] < N_E)]
ix_EI = ix[:, (ix[0] < N_E) & (ix[1] >= N_E)]
ix_II = ix[:, (ix[0] >= N_E) & (ix[1] >= N_E)]

covs_sim_EE = covs_sim[(ix_EE[0], ix_EE[1])]
covs_sim_EI = covs_sim[(ix_EI[0], ix_EI[1])]
covs_sim_II = covs_sim[(ix_II[0], ix_II[1])]
covs_thy_EE = covs_thy[(ix_EE[0], ix_EE[1])]
covs_thy_EI = covs_thy[(ix_EI[0], ix_EI[1])]
covs_thy_II = covs_thy[(ix_II[0], ix_II[1])]

corrs_sim_EE = corrs_sim[(ix_EE[0], ix_EE[1])]
corrs_sim_EI = corrs_sim[(ix_EI[0], ix_EI[1])]
corrs_sim_II = corrs_sim[(ix_II[0], ix_II[1])]
corrs_thy_EE = corrs_thy[(ix_EE[0], ix_EE[1])]
corrs_thy_EI = corrs_thy[(ix_EI[0], ix_EI[1])]
corrs_thy_II = corrs_thy[(ix_II[0], ix_II[1])]


###############################################################################
# Plotting of results
# -------------------
#
# Here we plot the firing rates, CVs, and pairwise covariances



###############################################################################
# options

blue = colors['blue']
red = colors['red']
yellow = colors['yellow']
neutral = colors['lightneutral']

alpha = 1
markerscale = 5

width = utils.mm2inch(180)
height = utils.mm2inch(60)

fig = plt.figure(figsize=[width, height], constrained_layout=True)

plt.rc('font', size=fontsize)
plt.rc('xtick', labelsize=labelsize)
plt.rc('ytick', labelsize=labelsize)

grid = (1, 3)

all_axs = []

# input_dict = np.load(plotting_data_path + network_id + '.npz',
#                         allow_pickle=True)
# network_params = input_dict['network_params'].tolist()
# N_E = network_params['N_E']

# results = input_dict['results'].tolist()

# rates_sim = results['rates_sim']
# corrs_sim_EE = results['corrs_sim_EE']
# corrs_sim_EI = results['corrs_sim_EI']
# corrs_sim_II = results['corrs_sim_II']
# covs_sim_EE = results['covs_sim_EE']
# covs_sim_EI = results['covs_sim_EI']
# covs_sim_II = results['covs_sim_II']

# rates_thy = results['rates_thy']
# corrs_thy_EE = results['corrs_thy_EE']
# corrs_thy_EI = results['corrs_thy_EI']
# corrs_thy_II = results['corrs_thy_II']
# covs_thy_EE = results['covs_thy_EE']
# covs_thy_EI = results['covs_thy_EI']
# covs_thy_II = results['covs_thy_II']

cc_rates = np.corrcoef(rates_sim, rates_thy)[0, 1]
cc_corrs = np.corrcoef(cross_corrs_sim, cross_corrs_thy)[0, 1]
cc_covs = np.corrcoef(cross_covs_sim, cross_covs_thy)[0, 1]

ax_rates = plt.subplot2grid(grid, (0, 0))
ax_covs = plt.subplot2grid(grid, (0, 1))
ax_corrs = plt.subplot2grid(grid, (0, 2))
axs = [ax_rates, ax_covs, ax_corrs]
all_axs.extend(axs)

plot_row(axs, rates_sim, rates_thy,
            covs_sim_EE, covs_sim_EI, covs_sim_II,
            covs_thy_EE, covs_thy_EI, covs_thy_II,
            corrs_sim_EE, corrs_sim_EI, corrs_sim_II,
            corrs_thy_EE, corrs_thy_EI, corrs_thy_II,
            cc_rates, cc_covs, cc_corrs,
            N_E)

all_axs[0].set_title('rate')
all_axs[1].set_title('covariance')
all_axs[2].set_title('correlation')

all_axs[0].set_ylabel('theory')

all_axs[0].set_xlabel('simulation')
all_axs[1].set_xlabel('simulation')
all_axs[2].set_xlabel('simulation')

all_axs[0].legend(markerscale=markerscale)
leg = all_axs[1].legend(markerscale=markerscale)
for lh in leg.legendHandles:
    lh.set_alpha(1)
leg = all_axs[2].legend(markerscale=markerscale)
for lh in leg.legendHandles:
    lh.set_alpha(1)

labels = list(string.ascii_lowercase[:len(all_axs)])
utils.add_panel_labels(all_axs, labels, fontsize=panel_label_fontsize,
                        use_parenthesis=True)
utils.remove_borders(all_axs)

plt.savefig(plot_path + network_id + '_thy_vs_sim.pdf', dpi=600)
plt.close()
