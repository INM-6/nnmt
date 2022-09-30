"""
Simulation and mean-field theory for multiple synaptic time constants
=====================================================================

.. image:: ../../../../examples/multiple_synaptic_time_constants/rasterplot.png
  :width: 1000
  :alt: Plot of simulated and estimated rates

Based on NEST 3.3
`random balanced network example <https://nest-simulator.readthedocs.io/en/v3.3/auto_examples/brunel_alpha_nest.html>`_
we simulate a network of leaky integrate-and-fire neurons with multiple
synaptic time constants and compare the results to the mean-field estimates
which use the effective synaptic time constant approach of
:cite:t:`fourcaud2002` (Eq. 5.49).
"""
import nnmt
import numpy as np
import scipy.special as sp

from nnmt import units

import nest
import nest.raster_plot
import matplotlib.pyplot as plt


###############################################################################
# Helper functions (not used at the moment):

def LambertWm1(x):
    # Using scipy to mimic the gsl_sf_lambert_Wm1 function.
    return sp.lambertw(x, k=-1 if x < 0 else 0).real


def ComputePSPnorm(tauMem, CMem, tauSyn):
    a = (tauMem / tauSyn)
    b = (1.0 / tauSyn - 1.0 / tauMem)

    # time of maximum
    t_max = 1.0 / b * (-LambertWm1(-np.exp(-1.0 / a) / a) - 1.0 / a)

    # maximum of PSP for current of unit amplitude
    return (np.exp(1.0) / (tauSyn * CMem * b) *
            ((np.exp(-t_max / tauMem) - np.exp(-t_max / tauSyn)) / b -
             t_max * np.exp(-t_max / tauSyn)))


###############################################################################
# Simulation parameters:

dt = 0.1    # the resolution in ms
T_init = 7000.0 # init time
T_sim = 2000.0  # Simulation time in ms


###############################################################################
# Network parameters:

g = 5.0  # ratio inhibitory weight/excitatory weight
eta = 3.0  # external rate relative to threshold rate
epsilon = 0.1  # connection probability

order = 1000
N_E = 4 * order  # number of excitatory neurons
N_I = 1 * order  # number of inhibitory neurons
N_neurons = N_E + N_I   # number of neurons in total
N_rec = 100      # record from 50 neurons

K_E = int(epsilon * N_E)  # number of excitatory synapses per neuron
K_I = int(epsilon * N_I)  # number of inhibitory synapses per neuron
K_tot = int(K_I + K_E)      # total number of synapses per neuron

theta = 20.0  # membrane threshold potential in mV
C_m = 250.0 # membrane capacitance in pF
J = 0.1   # postsynaptic amplitude in mV

delay = 1.5    # synapt ic delay in ms
tau_m = 20.0  # time constant of membrane potential in ms
tau_s_E = 0.5 # synaptic time constant in ms
tau_s_I = 0.8 # synaptic time constant in ms
tau_r = 2.0 # refractory time in ms

# convert synaptic weight from mV to pA
# J_unit_E = ComputePSPnorm(tau_m, C_m, tau_s_E) # factor for converting PSP to PSC
# J_unit_I = ComputePSPnorm(tau_m, C_m, tau_s_I) # factor for converting PSP to PSC
J_unit_E =  tau_s_E / C_m # factor for converting PSP to PSC
J_unit_I =  tau_s_I / C_m # factor for converting PSP to PSC
J_E = J / J_unit_E  # amplitude of excitatory postsynaptic current in pA
J_I = -g * J / J_unit_I  # amplitude of inhibitory postsynaptic current in pA

# compute Poisson rate
nu_th = (theta * C_m) / (J_E * K_E * np.exp(1) * tau_m * tau_s_E)
nu_ext = eta * nu_th
p_rate = 1000.0 * nu_ext * K_E

neuron_params = {"C_m": C_m,
                 "tau_m": tau_m,
                 "t_ref": tau_r,
                 "E_L": 0.0,
                 "V_reset": 0.0,
                 "V_m": 0.0,
                 "V_th": theta,
                 "tau_syn_ex": tau_s_E,
                 "tau_syn_in": tau_s_I}


###############################################################################
# Mean-field estimation:

# dict for nnmt
network_params = dict(
    populations = ['E', 'I'],
    K = np.array([[K_E, K_I],
                  [K_E, K_I]]),
    W = np.array([[J_E, J_I],
                  [J_E, J_I]]) * units.pA,
    tau_m = tau_m * units.ms,
    tau_s = np.array([tau_s_E, tau_s_I]) * units.ms,
    V_0_abs = 0 * units.mV,
    V_th_abs = theta * units.mV,
    W_ext = np.array([J_E, J_E]) * units.pA,
    K_ext = np.array([K_E, K_E]),
    tau_s_ext = tau_s_E * units.ms,
    tau_r = tau_r * units.ms,
    C = C_m * units.pF,
    nu_ext = nu_ext * units.kHz
)

# create nnmt model and compute firing rates
ei_network = nnmt.models.Basic(network_params)
rates = nnmt.lif.exp.firing_rates(ei_network)


###############################################################################
# NEST simulation:

nest.ResetKernel()
nest.resolution = dt
nest.print_time = True
nest.overwrite_files = True
nest.local_num_threads = 4

print("Building network")
nodes_ex = nest.Create("iaf_psc_exp", N_E, params=neuron_params)
nodes_in = nest.Create("iaf_psc_exp", N_I, params=neuron_params)
noise = nest.Create("poisson_generator", params={"rate": p_rate})

nest.CopyModel("static_synapse", "excitatory",
               {"weight": J_E, "delay": delay})
nest.CopyModel("static_synapse", "inhibitory",
               {"weight": J_I, "delay": delay})
syn_params_ex = {"synapse_model": "excitatory"}
syn_params_in = {"synapse_model": "inhibitory"}

print("Connecting noise")
nest.Connect(noise, nodes_ex, syn_spec=syn_params_ex)
nest.Connect(noise, nodes_in, syn_spec=syn_params_ex)

print("Connecting network")

print("Excitatory connections")

conn_params_ex = {'rule': 'fixed_indegree', 'indegree': K_E}
nest.Connect(nodes_ex, nodes_ex + nodes_in, conn_params_ex, syn_params_ex)

print("Inhibitory connections")

conn_params_in = {'rule': 'fixed_indegree', 'indegree': K_I}
nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_in, syn_params_in)

print("Warmup")

nest.Simulate(T_init)

print("Connecting recorders")

espikes = nest.Create("spike_recorder")
ispikes = nest.Create("spike_recorder")
# espikes.set(label="brunel-py-ex", record_to="ascii")
# ispikes.set(label="brunel-py-in", record_to="ascii")
nest.Connect(nodes_ex[:N_rec], espikes, syn_spec="excitatory")
nest.Connect(nodes_in[:N_rec], ispikes, syn_spec="excitatory")

print("Simulating")

nest.Simulate(T_sim)


###############################################################################
# Print results and plot simulation:

events_ex = espikes.n_events
events_in = ispikes.n_events

rate_ex = events_ex / T_sim * 1000.0 / N_rec
rate_in = events_in / T_sim * 1000.0 / N_rec

print(f'Mean-field: E: {rates[0]}; I: {rates[1]}')
print(f'Simulation: E: {rate_ex}; I: {rate_in}')

fig = nest.raster_plot.from_device(espikes, hist=True)
fig = plt.figure(1)
ax = fig.get_axes()[-1]
ax.axhline(rates[0], label='MFT', color='darkgrey', linestyle='dashed')
ax.legend()
plt.savefig('rasterplot.png', dpi=600)
plt.show()
