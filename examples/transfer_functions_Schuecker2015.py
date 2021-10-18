"""
Transfer Functions (Schuecker 2015)
===================================

Here we calculate the transfer functions as in :cite:t:`schuecker2015`.
"""

import nnmt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import matplotlib.ticker

plt.style.use('frontiers.mplstyle')
mpl.rcParams.update({'legend.fontsize': 'medium',  # old: 5.0 was too small
                     'axes.titlepad': 0.0,
                     })


# %%
# The parameters used for calculation of the transfer functions 
# in :cite:t:`schuecker2015` were gathered in a .yaml-File and are loaded here. 

params = nnmt.input_output.load_val_unit_dict_from_yaml(
        '../tests/fixtures/integration/config/Schuecker2015_parameters.yaml')

# without converting to si
network_params = params.copy()
nnmt.utils._strip_units(network_params)

# converting to si
si_network_params = params.copy()
nnmt.utils._convert_to_si_and_strip_units(si_network_params)

frequencies = np.logspace(
        si_network_params['f_start_exponent'],
        si_network_params['f_end_exponent'],
        si_network_params['n_freqs'])
# add the zero frequency
frequencies = np.insert(frequencies, 0, 0.0)
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
    
    transfer_function = nnmt.lif.exp._transfer_function_shift(
        si_network_params[f'mean_input_{index}'],
        si_network_params[f'sigma_{index}'],
        si_network_params['tau_m'],
        si_network_params['tau_s'],
        si_network_params['tau_r'],
        si_network_params['theta'],
        si_network_params['V_reset'],
        omegas,
        synaptic_filter=False)
    
    # the result is returned in SI-units (1/(s*V))
    # the original figure in the paper is in (1/(s*mV))
    transfer_function /= 1000
    
    # calculate properties plotted in Schuecker 2015
    absolute_value = np.abs(transfer_function)
    phase = (np.angle(transfer_function)
                / 2 / np.pi * 360)
    
    # collect all results
    absolute_values.append(absolute_value)
    phases.append(phase)
    nu0_fbs.append(nu0_fb)

# %% 
# Prepare data for plotting by parsing into a dictionary.
pre_results = dict(
    absolute_values=absolute_values,
    phases=phases,
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
                'nu0_fb': pre_results['nu0_fbs'][i][j]}
        
# %%
# Plotting
width = 3.34646 * 2
height = 3.34646 / 2
fig = plt.figure(figsize=(width, height),
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
        
        # shift the zero frequency to be plotted on log-scale
        zero_freq = 0.06
        
        if sigma == 4.0:
            ls = '-'
        else:
            ls = '--'

        firing_rate = round(test_results['sigma'][sigma]['mu'][mu]['nu0_fb'])
        # excluding zero frequency
        axA.semilogx(frequencies[1:],
                        test_results['sigma'][sigma]['mu'][mu]['absolute_value'][1:],
                        color=colors[i],
                        linestyle=ls,
                        linewidth=lw)
        axB.semilogx(frequencies,
                        test_results['sigma'][sigma]['mu'][mu]['phase'],
                        color=colors[i],
                        linestyle=ls,
                        linewidth=lw,
                        label=f'({np.round(mu, 1)}, {sigma})')
        # just zero frequency
        # axA.semilogx(zero_freq,
        #                 test_results['sigma'][sigma]['mu'][mu]['absolute_value'][0],
        #                 '+',
        #                 color=colors[i],
        #                 markersize=markersize_cross)

axA.set_xlabel(r'frequency $\omega/2\pi\quad(1/\mathrm{s})$')
axA.set_ylabel(r'amplitude $|n(\omega)|\quad(\mathrm{s}\cdot\mathrm{mV})^{-1}$'
               ,labelpad = 0)

axB.set_xlabel(r'frequency $\omega/2\pi\quad(1/\mathrm{s})$')
axB.set_ylabel(r'phase $\angle n(\omega)\quad(^{\circ})$'
               ,labelpad = 2)

axA.set_xticks([1e-1, 1e0, 1e1, 1e2])
axA.set_yticks([0, 6, 12])

axB.set_xticks([1e-1, 1e0, 1e1, 1e2])
axB.set_yticks([-60, -30, 0])

label_prms = dict(x=-0.3, y=1.2, fontsize=10, fontweight='bold',
                  va='top', ha='right')
axA.text(s='(A)', transform=axA.transAxes, **label_prms)
axB.text(s='(B)', transform=axB.transAxes, **label_prms)


x_minor = matplotlib.ticker.LogLocator(
    base = 10.0, 
    subs = np.arange(1.0, 10.0) * 0.1, 
    numticks = 10)
axA.xaxis.set_minor_locator(x_minor)
axA.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axB.xaxis.set_minor_locator(x_minor)
axB.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

axB.legend(title='$(\mu, \sigma)$ in mV', title_fontsize=None,
          handlelength=2, labelspacing=0.0)

plt.savefig('figures/transfer_functions_Schuecker2015.eps')