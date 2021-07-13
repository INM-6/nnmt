import nnmt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import matplotlib.ticker

plt.style.use('frontiers.mplstyle')

raw_params_with_units = nnmt.input_output.load_val_unit_dict_from_yaml(
        '../tests/fixtures/integration/config/Schuecker2015_parameters.yaml')
raw_params_with_units['dimension'] = 1

# without converting to si
params = raw_params_with_units.copy()
nnmt.utils._strip_units(params)
network_params = params

# converting to si
params = raw_params_with_units.copy()
nnmt.utils._to_si_units(params)
nnmt.utils._strip_units(params)
params['dimension'] = 1
si_network_params =  params 

frequencies = np.logspace(
        si_network_params['f_start_exponent'],
        si_network_params['f_end_exponent'],
        si_network_params['n_freqs'])

omegas = 2 * np.pi * frequencies

indices = [1,2]

# calculate nnmt results for different mus and sigmas
absolute_values = []
phases = []
zero_freqs = []
nu_0s = []
nu0_fbs = []
nu0_fb433s = []
for i, index in enumerate(indices):
    # Stationary firing rates for delta shaped PSCs.
    nu_0 = nnmt.lif.delta._firing_rates_for_given_input(
        si_network_params['V_reset'],
        si_network_params['theta'],
        si_network_params[f'mean_input_{index}'],
        si_network_params[f'sigma_{index}'],
        si_network_params['tau_m'],
        si_network_params['tau_r'])

    # Stationary firing rates for filtered synapses (via shift)
    nu0_fb = nnmt.lif.exp._firing_rate_shift(
        si_network_params['V_reset'],
        si_network_params['theta'],
        si_network_params[f'mean_input_{index}'],
        si_network_params[f'sigma_{index}'],
        si_network_params['tau_m'],
        si_network_params['tau_r'],
        si_network_params['tau_s'])

    # Stationary firing rates for exp PSCs. (via Taylor)
    nu0_fb433 = nnmt.lif.exp._firing_rate_taylor(
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
    zero_freq = transfer_function_zero_freq
    
    # collect all results
    absolute_values.append(absolute_value)
    phases.append(phase)
    zero_freqs.append(zero_freq)
    nu_0s.append(nu_0)
    nu0_fbs.append(nu0_fb)
    nu0_fb433s.append(nu0_fb433)

# prepare parsing into a dictionary
pre_results = dict(
    absolute_values=absolute_values,
    phases=phases,
    zero_freqs=zero_freqs,
    nu_0s=nu_0s,
    nu0_fbs=nu0_fbs,
    nu0_fb433s=nu0_fb433s)
    
# parse results into a dictionary
test_results = defaultdict(str)
test_results['sigma'] = defaultdict(dict)
for i, index in enumerate(indices):
    sigma = network_params[f'sigma_{index}']
    test_results['sigma'][sigma]['mu'] = (
        defaultdict(dict))
    for j, mu in enumerate(network_params[f'mean_input_{index}']):
        test_results[
            'sigma'][sigma][
            'mu'][mu] = {
                'absolute_value': pre_results['absolute_values'][i][:, j],
                'phase': pre_results['phases'][i][:, j],
                'zero_freq': pre_results['zero_freqs'][i][j],
                'nu_0': pre_results['nu_0s'][i][j],
                'nu0_fb': pre_results['nu0_fbs'][i][j],
                'nu0_fb433': pre_results['nu0_fb433s'][i][j]}
    
# plot
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
                        test_results['sigma'][sigma]['mu'][mu]['zero_freq'],
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
