"""
Transfer Functions (Schuecker 2015)
===================================

Here we calculate the transfer functions as in :cite:t:`schuecker2015`.
"""

import nnmt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import matplotlib.ticker

plt.style.use('frontiers.mplstyle')

# %%
# The parameters used for calculation of the transfer functions 
# in :cite:t:`schuecker2015` were gathered in a .yaml-File and are loaded here. 

raw_params_with_units = nnmt.input_output.load_val_unit_dict_from_yaml(
        '../tests/fixtures/integration/config/Schuecker2015_parameters.yaml')
raw_params_with_units['dimension'] = 1

# without converting to si
params = raw_params_with_units.copy()
nnmt.utils._strip_units(params)
network_params = params

# converting to si
si_network_params = raw_params_with_units.copy()
nnmt.utils._to_si_units(si_network_params)
nnmt.utils._strip_units(si_network_params)
si_network_params['dimension'] = 1

frequencies = np.logspace(
        si_network_params['f_start_exponent'],
        si_network_params['f_end_exponent'],
        si_network_params['n_freqs'])

omegas = 2 * np.pi * frequencies

indices = [1,2]

# %%
# Calculate results for different input means and standard deviations.

absolute_values = []
phases = []
transfer_function_zero_freqs = []
nu0_fbs = []
for i, index in enumerate(indices):
    # Stationary firing rates for filtered synapses (via shift)
    nu0_fb = nnmt.lif.exp._firing_rate_shift(
        si_network_params['V_reset'],
        si_network_params['theta'],
        si_network_params[f'mean_input_{index}'],
        si_network_params[f'sigma_{index}'],
        si_network_params['tau_m'],
        si_network_params['tau_r'],
        si_network_params['tau_s'])
    
    # colored noise zero-frequency limit of transfer function
    transfer_function_zero_freq = (
        nnmt.lif.exp._derivative_of_firing_rates_wrt_mean_input(
            si_network_params['V_reset'],
            si_network_params['theta'],
            si_network_params[f'mean_input_{index}'],
            si_network_params[f'sigma_{index}'],
            si_network_params['tau_m'],
            si_network_params['tau_r'],
            si_network_params['tau_s'])) / 1000
    
    transfer_function = nnmt.lif.exp._transfer_function_shift(
        si_network_params[f'mean_input_{index}'],
        si_network_params[f'sigma_{index}'],
        si_network_params['tau_m'],
        si_network_params['tau_s'],
        si_network_params['tau_r'],
        si_network_params['theta'],
        si_network_params['V_reset'],
        omegas,
        synaptic_filter=False) / 1000
    
    # calculate properties plotted in Schuecker 2015
    absolute_value = np.abs(transfer_function)
    phase = (np.angle(transfer_function)
                / 2 / np.pi * 360)
    
    # collect all results
    absolute_values.append(absolute_value)
    phases.append(phase)
    transfer_function_zero_freqs.append(transfer_function_zero_freq)
    nu0_fbs.append(nu0_fb)

# %% 
# Prepare data for plotting by parsing into a dictionary.
pre_results = dict(
    absolute_values=absolute_values,
    phases=phases,
    transfer_function_zero_freqs=transfer_function_zero_freqs,
    nu0_fbs=nu0_fbs)
    
test_results = defaultdict(str)
test_results['sigma'] = defaultdict(dict)
for i, index in enumerate(indices):
    sigma = network_params[f'sigma_{index}']
    test_results['sigma'][sigma]['mu'] = (
        defaultdict(dict))
    for j, mu in enumerate(network_params[f'mean_input_{index}']):
        test_results['sigma'][sigma]['mu'][mu] = {
                'absolute_value': pre_results['absolute_values'][i][:, j],
                'phase': pre_results['phases'][i][:, j],
                'transfer_function_zero_freq': \
                    pre_results['transfer_function_zero_freqs'][i][j],
                'nu0_fb': pre_results['nu0_fbs'][i][j]}
        
# %%
# Plotting
fig = plt.figure(figsize=(3.34646, 3.34646/2),
                 constrained_layout=True)

grid_specification = gridspec.GridSpec(1, 2, figure=fig)

axA = fig.add_subplot(grid_specification[0])
axB = ax = fig.add_subplot(grid_specification[1])

for sigma in test_results['sigma'].keys():
    for i, mu in enumerate(test_results['sigma'][sigma]['mu'].keys()):
      
        print(sigma)
        colors = ['black', 'grey']
        lw = 1
        markersize_cross = 4
        
        zero_freq = 0.06
        
        if sigma == 4.0:
            ls = '-'
        else:
            ls = '--'

        axA.semilogx(frequencies,
                        test_results['sigma'][sigma]['mu'][mu]['absolute_value'],
                        color=colors[i],
                        linestyle=ls,
                        linewidth=lw,
                        label=r'$\nu=$ {} Hz'.format(nu0_fb))
        axB.semilogx(frequencies,
                        test_results['sigma'][sigma]['mu'][mu]['phase'],
                        color=colors[i],
                        linestyle=ls,
                        linewidth=lw,
                        label=r'$\mu=$ ' + str(mu))
        axA.semilogx(zero_freq,
                        test_results['sigma'][sigma]['mu'][mu]['transfer_function_zero_freq'],
                        '+',
                        color=colors[i],
                        markersize=markersize_cross)

axA.set_xlabel(r'frequency $\omega/2\pi\quad(1/\mathrm{s})$')
axA.set_ylabel(r'$|\frac{n(\omega)\nu}{\epsilon\mu}|\quad(\mathrm{s}\,\mathrm{mV})^{-1}$',labelpad = 0)

axB.set_xlabel(r'frequency $\omega/2\pi\quad(1/\mathrm{s})$')
axB.set_ylabel(r'$-\angle n(\omega)\quad(^{\circ})$',labelpad = 2)

axA.set_xticks([1e-1, 1e0, 1e1, 1e2])
axA.set_yticks([0, 6, 12])

axB.set_xticks([1e-1, 1e0, 1e1, 1e2])
axB.set_yticks([-60, -30, 0])

x_minor = matplotlib.ticker.LogLocator(
    base = 10.0, 
    subs = np.arange(1.0, 10.0) * 0.1, 
    numticks = 10)
axA.xaxis.set_minor_locator(x_minor)
axA.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axB.xaxis.set_minor_locator(x_minor)
axB.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

plt.savefig('Schuecker_Fig4.png')