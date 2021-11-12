"""
Fitting the transfer function for different working points
==========================================================

This example demonstrates the methods used in Figure 5 of :cite:t:`senk2020`.
A figure illustrating the network structure of the used model is set up in
:doc:`network_structure`.
The same model is used in the example :doc:`spatial_patterns`.

"""

import nnmt.lif.exp as mft  # main set of meanfield tools
from nnmt.models.basic import Basic as BasicNetwork
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import ticker
plt.style.use('frontiers.mplstyle')
mpl.rcParams.update({'legend.fontsize': 'medium',  # old: 5.0 was too small
                     'axes.titlepad': 0.0})

###############################################################################
# First, we define parameters for data generation and plotting.

params = {
    # mean and standard deviations of inputs to scan (in V)
    'mean_inputs_scan': np.arange(6., 14., 2.) * 1e-3,
    'std_inputs_scan': np.arange(6., 14., 2.) * 1e-3,

    # pairs of mean and standard deviation of inputs to show transfer function
    # and fit (in V)
    'mean_std_inputs_tf': np.array([[6., 6.],
                                    [6., 12.],
                                    [10., 10.],
                                    [12., 12.]]) * 1e-3,

    # mean and standard deviation of input used for stability analysis (in V)
    'mean_std_inputs_stability': np.array([10., 10.]) * 1e-3,

    # figure width in inch
    'figwidth_2cols': 180. / 25.4,

    # labels and corresponding scaling parameters for plotted quantities
    'quantities': {
        'mean_input': {
            'label': r'mean input $\mu$ (mV)',
            'scale': 1e3},
        'std_input': {
            'label': r'std input $\sigma$ (mV)',
            'scale': 1e3},
        'nu_ext_exc': {
            'label': 'exc. external rate\n' + r'$\nu_\mathrm{ext,E}$ (1000/s)',
            'scale': 1e-3},
        'nu_ext_inh': {
            'label': 'inh. external rate\n' + r'$\nu_\mathrm{ext,I}$ (1000/s)',
            'scale': 1e-3},
        'firing_rates': {
            'label': 'rate\n' + r'$\nu$ (1/s)',
            'scale': 1.},
        'tau_rate': {
            'label': 'fit time constant\n' + r'$\tau$ (ms)',
            'scale': 1e3},
        'W_rate': {
            'label': 'fit exc. weight\n' + r'$w_\mathrm{E}$',
            'scale': 1.},  # unitless
        'fit_error': {
            'label': 'fit error\n' + r'$\epsilon$ (%)',
            'scale': 1e2},
        'transfer_function': {
            'label': r'transfer function $H_\mu$',
            'scale': 1e-3},
        'transfer_function_amplitude': {
            'label':
                r'amplitude $|H_\mu|\quad(\mathrm{s}\cdot\mathrm{mV})^{-1}$'},
        'transfer_function_phase': {
            'label': r'phase $\angle H\mu\quad(\circ)$', },
        'frequencies': {
            'label': r'frequency $\mathrm{Im}[\lambda]/(2\pi)$ (Hz)',
            'scale': 1.}},

    # color definitions
    # numbers from discrete rainbow scheme of https://personal.sron.nl/~pault
    'colors': {
        'light_grey': '#BBBBBB',
        'dark_grey': '#555555',
        'dark_purple': '#882E72',  # no. 9
        'light_purple': '#D1BBD7',  # no. 3
        'dark_green': '#4EB265',  # no. 15
        'light_green': '#CAE0AB',  # no. 17
        'dark_orange': '#E8601C',  # no. 24
        'light_orange': '#F6C141'},  # no. 20

    # colors for transfer function [dark for LIF trans. func., light for fit]
    'colors_tf': [
        ['dark_purple', 'light_purple'],
        ['dark_green', 'light_green'],
        ['dark_orange', 'light_orange'],
        ['dark_grey', 'light_grey']]}


###############################################################################
# We also define a helper function for adding labels to figure panels.

def _add_label(ax, label, xshift=0., yshift=0., scale_fs=1.):
    """
    Adds label to plot panel given by axis.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes object
        Axes.
    label : str
        Letter.
    xshift : float
        x-shift of label position.
    yshift : float
        y-shift of label position.
    scale_fs : float
        Scale factor for font size.
    """
    label_pos = [0., 1.]
    ax.text(label_pos[0] + xshift, label_pos[1] + yshift, '(' + label + ')',
            ha='left', va='bottom',
            transform=ax.transAxes, fontweight='bold',
            fontsize=mpl.rcParams['font.size'] * scale_fs)

##########################################################################
# ===============
# Generating data
# ===============
# We instantiate a ``Basic`` model with a set of pre-defined network and
# analysis parameters.
# The relative inhibition is here g = 5 in contrast to the original Figure 5 of
# :cite:t:`senk2020` which uses g = 6.


network = BasicNetwork(
    network_params='Senk2020_network_params.yaml',
    analysis_params='Senk2020_analysis_params.yaml')

##########################################################################
# All results will be stored in ``tf_scan_results``.

tf_scan_results = {}
tf_scan_results['frequencies'] = \
    network.analysis_params['omegas'] / (2. * np.pi)
dims = (len(params['mean_inputs_scan']), len(params['std_inputs_scan']))
for key in ['nu_ext_exc', 'nu_ext_inh', 'firing_rates',
            'tau_rate', 'fit_error', 'W_rate']:
    tf_scan_results[key] = np.zeros(dims)
for key in ['transfer_function', 'transfer_function_fit']:
    tf_scan_results[key] = np.zeros(
        (dims[0], dims[1], len(network.analysis_params['omegas'])),
        dtype=complex)

##########################################################################
# The main loop for generating the data iterates over working points which are
# defined as pairs of mean and standard deviation of inputs.
# For each working point, we first compute the excitatory and inhibitory
# external firing rates required to preserve the working point and adjust the
# network parameters accordingly.
# Then, we calculate the LIF transfer function and fit it with the one of a
# low-pass filter using a least-squares fit.

print('Iterating over working points and fitting the LIF transfer function.')

for i, mu in enumerate(params['mean_inputs_scan']):
    for j, sigma in enumerate(params['std_inputs_scan']):

        print(f'    (mu, sigma) = ({mu * 1e3}, {sigma * 1e3}) mV')

        # fix working point via external rates
        nu_ext = mft.external_rates_for_fixed_input(
            network, mu_set=mu, sigma_set=sigma)

        network.change_parameters(
            changed_network_params={'nu_ext': nu_ext},
            overwrite=True)

        # calculate transfer function and its fit
        mft.working_point(network)
        mft.transfer_function(network)
        mft.fit_transfer_function(network)

        # store results
        tf_scan_results['nu_ext_exc'][i, j] = nu_ext[0]
        tf_scan_results['nu_ext_inh'][i, j] = nu_ext[1]

        # 1D results (assert equal values for populations, store only one)
        for key in ['firing_rates', 'tau_rate', 'fit_error']:
            res = network.results[mft._prefix + key]
            assert len(np.shape(res)) == 1 and len(np.unique(res)) == 1
            tf_scan_results[key][i, j] = res[0]

        # 2D results (assert equal rows, store only first value (E->E,I))
        for key in ['W_rate']:
            res = network.results[mft._prefix + key]
            assert len(
                np.shape(res)) == 2 and np.isclose(
                res, res[0]).all()
            tf_scan_results[key][i, j] = res[0, 0]

        # 2D results (assert equal columns for populations, store only one)
        for key in ['transfer_function', 'transfer_function_fit']:
            res = network.results[mft._prefix + key]
            res_t = np.transpose(res)
            assert (len(np.shape(res)) == 2) and (
                np.isclose(res_t, res_t[0]).all())
            tf_scan_results[key][i, j] = res[:, 0]

##########################################################################
# ========
# Plotting
# ========
# We generate a figure with three panels to show the results from scanning
# over the input.
# The figure spans two columns.

print('Plotting.')

fig = plt.figure(figsize=(params['figwidth_2cols'],
                          params['figwidth_2cols'] / 2))
gs = gridspec.GridSpec(1, 10, figure=fig)

##########################################################################
# First, we plot results from scanning over the full ranges of working points.
# Panel A contains the fixed external rates and the predicted firing rates
# of the neuronal populations.
# Panel C contains the results from fitting the transfer function, i.e.,
# the time constants, weights, and fit errors.

gs_wp = gridspec.GridSpecFromSubplotSpec(
    2, 3, subplot_spec=gs[0, :6], hspace=0.3, wspace=0)

mus = params['mean_inputs_scan']  # first index
sigmas = params['std_inputs_scan']  # second index
mu_star = params['mean_std_inputs_stability'][0]
sigma_star = params['mean_std_inputs_stability'][1]

for k, key in enumerate([
    'nu_ext_exc', 'nu_ext_inh', 'firing_rates',  # panel A
        'tau_rate', 'W_rate', 'fit_error']):  # panel C
    ax = plt.subplot(gs_wp[k])
    img = ax.pcolormesh(
        np.transpose(
            tf_scan_results[key] *
            params['quantities'][key]['scale']))

    # pcolormesh places ticks by default to lower bound, therefore add 0.5
    ax.set_xticks(np.arange(len(mus)) + 0.5)
    ax.set_yticks(np.arange(len(sigmas)) + 0.5)
    ax.set_xticklabels(
        (mus * params['quantities']['mean_input']['scale']).astype(int))
    ax.set_yticklabels(
        (sigmas * params['quantities']['std_input']['scale']).astype(int))

    if k == 1 or k == 4:
        ax.set_xlabel(params['quantities']['mean_input']['label'])

    if k == 0 or k == 3:
        ax.set_ylabel(params['quantities']['std_input']['label'])
    else:
        ax.set_yticklabels([])

    xshift = -0.6
    yshift = 0.22
    if k == 0:
        _add_label(ax, 'A', xshift=xshift, yshift=yshift)
    if k == 3:
        _add_label(ax, 'C', xshift=xshift, yshift=yshift)

    cb = plt.colorbar(img)
    cb.ax.tick_params(pad=0)
    cb.locator = ticker.MaxNLocator(nbins=4)
    cb.update_ticks()

    # star for mu and sigma used in this circuit (0.5 offset for
    # pcolormesh)
    xmu = (np.max(ax.get_xticks() - 0.5) * (mu_star - np.min(mus))
           / (np.max(mus) - np.min(mus)) + 0.5)
    ysigma = (np.max(ax.get_yticks() - 0.5)
              * (sigma_star - np.min(sigmas))
              / (np.max(sigmas) - np.min(sigmas)) + 0.5)
    ax.plot(xmu, ysigma,
            marker='*', markerfacecolor='white', markeredgecolor='none',
            markersize=mpl.rcParams['lines.markersize'] * 2.5)
    ax.plot(xmu, ysigma,
            marker='*', markerfacecolor='k', markeredgecolor='none',
            markersize=mpl.rcParams['lines.markersize'] * 2.)

    ax.set_title(params['quantities'][key]['label'])

##########################################################################
# To panel B, we plot the LIF transfer function and its fit for some selected
# working points.

gs_tf = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs[0, 7:], hspace=0)
ax_amplitude = plt.subplot(gs_tf[0])
_add_label(ax_amplitude, 'B', xshift=-0.4, yshift=0.02)
ax_phase = plt.subplot(gs_tf[1])

leg_handles, leg_labels = [[], []], []
c = 0
for i, mu in enumerate(params['mean_inputs_scan']):
    for j, sigma in enumerate(params['std_inputs_scan']):
        if [mu, sigma] in params['mean_std_inputs_tf'].tolist():
            cols = [params['colors'][x] for x in params['colors_tf'][c]]

            transfer_function = (
                tf_scan_results['transfer_function'][i, j]
                * params['quantities']['transfer_function']['scale'])

            transfer_function_fit = (
                tf_scan_results['transfer_function_fit'][i, j]
                * params['quantities']['transfer_function']['scale'])

            frequencies = (
                tf_scan_results['frequencies']
                * params['quantities']['frequencies']['scale'])

            # amplitude
            tf_orig = np.abs(transfer_function)
            tf_fit = np.abs(transfer_function_fit)
            ax_amplitude.plot(frequencies, tf_orig, c=cols[0])
            ax_amplitude.plot(
                frequencies, tf_fit,
                c=cols[1], linestyle='none', marker='o',
                markersize=mpl.rcParams['lines.markersize'] * 0.1)
            ax_amplitude.set_title(
                params['quantities']['transfer_function']['label'])

            # phase
            tf_orig = np.arctan2(np.imag(transfer_function),
                                 np.real(transfer_function)) * 180 / np.pi
            tf_fit = np.arctan2(
                np.imag(transfer_function_fit),
                np.real(transfer_function_fit)) * 180 / np.pi
            ax_phase.plot(frequencies, tf_orig, c=cols[0])
            ax_phase.plot(
                frequencies, tf_fit,
                c=cols[1], linestyle='none', marker='o',
                markersize=mpl.rcParams['lines.markersize'] * 0.1)
            ax_phase.set_xlabel(
                params['quantities']['frequencies']['label'])

            for ax, ylabel in zip(
                [ax_amplitude, ax_phase],
                [params['quantities']['transfer_function_amplitude']['label'],
                 params['quantities']['transfer_function_phase']['label']]):

                if any(frequencies > 0):
                    ax.set_xscale('log')
                ax.set_ylabel(ylabel)
                ax.set_xlim(frequencies[0], frequencies[-1])
            ax_amplitude.set_xticklabels([])

            leg_handles[0].append(Patch(facecolor=cols[0]))
            leg_handles[1].append(Patch(facecolor=cols[1]))
            leg_labels.append(f'({int(mu * 1e3)}, {int(sigma * 1e3)})')
            c += 1

##########################################################################
# For panel B, we customize a legend.

leg_handles = leg_handles[0] + leg_handles[1]
leg_labels = [''] * len(leg_labels) + leg_labels
ax_phase.legend(
    handles=leg_handles,
    labels=leg_labels,
    title=r'$(\mu, \sigma)$ in mV',
    ncol=2,
    handletextpad=0.5,
    handlelength=1.,
    columnspacing=-0.5)

spiking = mpl.lines.Line2D([], [], color='k', label='spiking model')
rate = mpl.lines.Line2D(
    [], [], color='k', linestyle='none', marker='o',
    markersize=mpl.rcParams['lines.markersize'] * 0.1,
    label='rate model (fit)')
ax_amplitude.legend(handles=[spiking, rate])

##########################################################################
# The final figure is saved to file.

plt.savefig('Senk2020_fit_transfer_function.eps')
