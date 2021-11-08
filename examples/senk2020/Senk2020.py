"""
Senk et al. 2020
================

Example demonstrating the methods used in Figures 5, 6 of :cite:t:`senk2020`.

Author: Johanna Senk
"""

##########################################################################
# Executing this script first generates data using the functions
# ``scan_fit_transfer_function()`` and ``linear_stability_analysis()``.
# This data is written to ``.npy`` files in the ``temp`` directory.
# Each of the following three functions generates one figure:
# ``network sketches``, results from ``scan_fit_transfer_function()``, and
# results from ``linear_stability_analysis()``.

import os
import sys
import nnmt.spatial as spatial
import nnmt.linear_stability as linstab
import nnmt.lif.exp as mft  # main set of meanfield tools
from nnmt.models.basic import Basic as BasicNetwork
import numpy as np
import scipy.optimize as sopt
import scipy.integrate as sint
import scipy.misc as smisc
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from matplotlib import ticker
plt.style.use('frontiers.mplstyle')
mpl.rcParams.update({'legend.fontsize': 'medium',  # old: 5.0 was too small
                     'axes.titlepad': 0.0,
                     })
try:
    import svgutils.transform as sg
except BaseException:
    pass


##########################################################################
# =====================
# Parameter definitions
# =====================
params = {
    # figure width in inch
    'figwidth_1col': 85. / 25.4,
    'figwidth_2cols': 180. / 25.4,

    # file name of final figure
    'figure_fname': 'Senk2020',

    # file names for intermediate results
    'fname_tf_scan_results': 'temp/Senk2020_scan_fit_transfer_function.npy',
    'fname_stability_results': 'temp/Senk2020_stability.npy',

    # labels and corresponding scaling parameters for plotted quantities
    'quantities': {
        'displacement': {'label': 'displacement $d$ (mm)',
                         'scale': 1e3},
        'mean_input': {'label': r'mean input $\mu$ (mV)',
                       'scale': 1e3},
        'std_input': {'label': r'std input $\sigma$ (mV)',
                      'scale': 1e3},
        'nu_ext_exc': {
            'label': 'exc. external rate\n' + r'$\nu_\mathrm{ext,E}$ (1000/s)',
            'scale': 1e-3},
        'nu_ext_inh': {
            'label': 'inh. external rate\n' + r'$\nu_\mathrm{ext,I}$ (1000/s)',
            'scale': 1e-3},
        'firing_rates': {'label': 'rate\n' + r'$\nu$ (1/s)',
                         'scale': 1.},
        'tau_rate': {'label': 'fit time constant\n' + r'$\tau$ (ms)',
                     'scale': 1e3},
        'W_rate': {'label': 'fit exc. weight\n' + r'$w_\mathrm{E}$',
                   'scale': 1.},  # unitless
        'fit_error': {'label': 'fit error\n' + r'$\epsilon$ (%)',
                      'scale': 1e2},
        'transfer_function': {'label': 'transfer function $H_\mu$',
                              'scale': 1e-3},
        'transfer_function_amplitude': {
            'label':
                r'amplitude $|H_\mu|\quad(\mathrm{s}\cdot\mathrm{mV})^{-1}$'},
        'transfer_function_phase': {
            'label': r'phase $\angle H\mu\quad(\circ)$', },
        'frequencies': {
            'label': r'frequency $\mathrm{Im}[\lambda]/(2\pi)$ (Hz)',
            'scale': 1.},
        'k_wavenumbers': {'label': 'wavenumber $k$ (1/mm)',
                          'scale': 1e-3},
        'eigenvalues': {'label': r'eigenvalue $\lambda$'},
        'eigenvalues_real': {'label': 'Re[$\lambda$] (1000/s)',
                             'scale': 1e-3},
        'eigenvalues_imag': {'label': 'Im[$\lambda$] (1000/s)',
                             'scale': 1e-3},
    },

    # generic set of colors
    'colors': {
        'ex_blue': '#4C72B0',
        'inh_red': '#C44E52',
        'light_yellow': '#EECC66',
        'dark_yellow': '#997700',
        'light_red': '#EE99AA',
        'dark_red': '#994455',
        'light_blue': '#6699CC',
        'dark_blue': '#004488',
        'light_grey': '#BBBBBB',
        'dark_grey': '#555555',
        'dark_purple': '#882E72',  # no. 9
        'light_purple': '#D1BBD7',  # no. 3
        'dark_green': '#4EB265',  # no. 15
        'light_green': '#CAE0AB',  # no. 17
        'dark_orange': '#E8601C',  # no. 24
        'light_orange': '#F6C141',  # no. 20
    },


    # mean and standard deviations of inputs to scan (in V)
    'mean_inputs_scan': np.arange(6., 14., 2.) * 1e-3,
    'std_inputs_scan': np.arange(6., 14., 2.) * 1e-3,

    # pairs of mean and standard deviation of inputs to show transfer function
    # and fit (in V)
    'mean_std_inputs_tf': np.array([[6., 6.],
                                    [6., 12.],
                                    [10., 10.],
                                    [12., 12.]]) * 1e-3,

    # colors for transfer function [dark for LIF trans. func., light for fit]
    'colors_tf': [['dark_purple', 'light_purple'],
                  ['dark_green', 'light_green'],
                  ['dark_orange', 'light_orange'],
                  ['dark_grey', 'light_grey']],

    # mean and standard deviation of input used for stability analysis (in V)
    'mean_std_inputs_stability': np.array([10., 10.]) * 1e-3,

    # colors for branches of Lambert W function
    'colors_br': {0: 'dark_red',
                  -1: 'light_red',
                  1: 'dark_blue',
                  -2: 'light_blue',
                  2: 'dark_yellow',
                  -3: 'light_yellow'},
}


##########################################################################
# ==============
# Main functions
# ==============
#
# Calculate results (meanfield and linear stability analysis)
# and create the figure.
def scan_fit_transfer_function():
    """
    Iterates over working points and fits the LIF transfer function.

    Iterates over pairs of mean and standard deviations of the input to compute
    - the excitatory and inhibitory external firing rates required to preserve
      the working point,
    - the firing rates of the neuronal populations,
    - the LIF transfer function, and
    - the least-squares fit of the LIF transfer function (low-pass filter) with
      the fitting error and the fit results (time constants and weights).
    """
    print('Iterating over working points and fitting the LIF transfer '
          'function.')

    network = BasicNetwork(
        network_params='parameters/Senk2020_network_params.yaml',
        analysis_params='parameters/Senk2020_analysis_params.yaml')

    dims = (len(params['mean_inputs_scan']), len(params['std_inputs_scan']))
    tf_scan_results = {}
    tf_scan_results['frequencies'] = \
        network.analysis_params['omegas'] / (2. * np.pi)
    for key in ['nu_ext_exc', 'nu_ext_inh',
                'firing_rates', 'tau_rate', 'fit_error',
                'W_rate']:
        tf_scan_results[key] = np.zeros(dims)
    for key in ['transfer_function', 'transfer_function_fit']:
        tf_scan_results[key] = np.zeros(
            (dims[0], dims[1], len(network.analysis_params['omegas'])),
            dtype=complex)

    # scan over working points (mean and standard deviation of inputs)
    for i, mu in enumerate(params['mean_inputs_scan']):
        for j, sigma in enumerate(params['std_inputs_scan']):
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
    np.save(params['fname_tf_scan_results'], tf_scan_results)


def linear_stability_analysis():
    """
    Performs linear stability analysis for the rate and  spiking models.

    Fixes the working point, computes and fits the LIF transfer function and
    assess the linear stability of the network with a spatial connectivity
    profile.
    Analytical solution for eigenvalues as a function of wavenumbers for
    branches of the Lambert W function with rate model parameters obtained by
    fitting the LIF transfer function.
    Linear interpolation of eigenvalues from rate model (analytical) to spiking
    model (LIF, only numerical) by two methods:
    - solving the full characteristic equation numerically and
    - integrating the derivative of the eigenvalue with respect to the
      interpolation parameter.
    """
    print('Performing linear stability analysis '
          'for the rate and spiking models.')

    network = BasicNetwork(
        network_params='parameters/Senk2020_network_params.yaml',
        analysis_params='parameters/Senk2020_analysis_params.yaml')

    # fix working point via external rates
    nu_ext = mft.external_rates_for_fixed_input(
        network,
        mu_set=params['mean_std_inputs_stability'][0],
        sigma_set=params['mean_std_inputs_stability'][1])

    network.change_parameters(changed_network_params={'nu_ext': nu_ext},
                              overwrite=True)

    # calculate transfer function and its fit
    mft.working_point(network)
    mft.transfer_function(network)
    mft.fit_transfer_function(network)

    # fit results
    tau_rate = network.results[mft._prefix + 'tau_rate']
    W_rate = network.results[mft._prefix + 'W_rate']

    # linear stability analysis
    branches = sorted(params['colors_br'].keys())
    k_wavenumbers = network.analysis_params['k_wavenumbers']
    eigenvalues = np.zeros((len(branches), len(k_wavenumbers)), dtype=complex)
    for i, branch_nr in enumerate(branches):
        for j, k_wavenumber in enumerate(k_wavenumbers):
            connectivity = W_rate * spatial._ft_spatial_profile_boxcar(
                k_wavenumber, network.network_params['width'])
            eigenvalues[i, j] = (
                linstab._solve_chareq_lambertw_constant_delay(
                    branch_nr=branch_nr, tau=tau_rate,
                    delay=network.network_params['D_mean'],
                    connectivity=connectivity))
    # index of eigenvalue with maximum real part
    idx_max = list(
        np.unravel_index(np.argmax(eigenvalues.real), eigenvalues.shape))

    # if max at branch -1, swap with 0
    if branches[idx_max[0]] == -1:
        idx_n1 = idx_max[0]  # index of current branch -1
        idx_0 = list(branches).index(0)  # index of current branch 0
        eigenvalues[[idx_n1, idx_0], [idx_0, idx_n1]]
        idx_max[0] = idx_0

    eigenval_max = eigenvalues[idx_max[0], idx_max[1]]
    k_eigenval_max = k_wavenumbers[idx_max[1]]
    idx_k_eigenval_max = idx_max[1]

    # linear interpolation
    alphas = np.linspace(0, 1, 5)
    lambdas_integral = np.zeros((len(branches), (len(alphas))), dtype=complex)
    lambdas_chareq = np.zeros((len(branches), len(alphas)), dtype=complex)
    for i, branch_nr in enumerate(branches):
        # evaluate all eigenvalues at k_eig_max
        # (wavenumbers with largest real part of eigenvalue from theory)
        lambda0 = eigenvalues[i, idx_k_eigenval_max]
        # 1. solution by solving the characteristic equation numerically
        for j, alpha in enumerate(alphas):
            lambdas_chareq[i, j] = _solve_chareq_numerically_alpha(
                lambda_rate=lambda0, k=k_eigenval_max, alpha=alpha,
                network=network, tau_rate=tau_rate, W_rate=W_rate)

        # 2. solution by solving the integral
        lambdas_integral[i, :] = _solve_lambda_of_alpha_integral(
            lambda_rate=lambda0, k=k_eigenval_max, alphas=alphas,
            network=network, tau_rate=tau_rate, W_rate=W_rate)

    stability_results = {
        'branches': branches,
        'k_wavenumbers': k_wavenumbers,
        'eigenvalues': eigenvalues,
        'eigenval_max': eigenval_max,
        'k_eigenval_max': k_eigenval_max,
        'idx_k_eigenval_max': idx_k_eigenval_max,
        'alphas': alphas,
        'lambdas_integral': lambdas_integral,
        'lambdas_chareq': lambdas_chareq}

    np.save(params['fname_stability_results'], stability_results)


def figure_Senk2020_network_structure():
    """
    Illustrates network structure.

    (A) Populations and external input.
    (B) Ring network illustration.
    (C) Spatial profile (boxcar connectivity kernel).
    """
    assert 'svgutils' in sys.modules, (
        'This figure requires: "import svgutils.transform as sg"')

    plot_fn = params['figure_fname'] + '_network_structure'
    sketch_fn = 'Senk2020_sketch.svg'

    fig = plt.figure(
        figsize=(
            params['figwidth_1col'],
            params['figwidth_1col']))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.2, hspace=0)

    _add_label(plt.subplot(gs[0, :]), 'A', xshift=-0.04, yshift=-0.1)
    plt.gca().set_axis_off()

    ax = _plot_network_sketch_sun(gs[1, 0])
    _add_label(ax, 'B', xshift=-0.085)

    ax = _plot_spatial_profile(gs[1, 1])
    _add_label(ax, 'C', yshift=1.08)

    svg_mpl = sg.from_mpl(fig, savefig_kw=dict(transparent=True))
    w_svg, h_svg = svg_mpl.get_size()
    svg_mpl.set_size((w_svg + 'pt', h_svg + 'pt'))
    svg_sketch = sg.fromfile(sketch_fn).getroot()
    svg_sketch.moveto(x=50, y=10, scale_x=1.5)
    svg_mpl.append(svg_sketch)
    svg_mpl.save(f'{plot_fn}.svg')
    os_return = os.system(f'inkscape --export-eps={plot_fn}.eps {plot_fn}.svg')
    if os_return == 0:
        os.remove(f'{plot_fn}.svg')
    else:
        print('Conversion to eps using inkscape failed, keeping svg...')


def figure_Senk2020_input_scan():
    """
    Loads and plots precomputed results from scanning working ponits.

    (A) Input scan: set external rates and predicted rates.
    (B) Transfer function (original and fit).
    (C) Input scan: fit results.
    """
    tf_scan_results = np.load(params['fname_tf_scan_results'],
                              allow_pickle=True).item()

    fig = plt.figure(figsize=(params['figwidth_2cols'],
                              params['figwidth_2cols'] / 2))
    gs = gridspec.GridSpec(1, 10, figure=fig)

    axes = _plot_mean_std_images(
        gs[0, :6], tf_scan_results)
    xshift = -0.6
    yshift = 0.22
    _add_label(axes[0], 'A', xshift=xshift, yshift=yshift)
    _add_label(axes[3], 'C', xshift=xshift, yshift=yshift)

    ax = _plot_transfer_functions(gs[0, 7:], tf_scan_results)
    _add_label(ax, 'B', xshift=-0.4, yshift=0.02)

    plt.savefig(params['figure_fname'] + '_input_scan.eps')


def figure_Senk2020_eigenvalues():
    """
    Loads and plots precomputed results from linear stability analysis.

    (A) Eigenvalues vs. wavenumbers.
    (B) Eigenvalues vs. interpolation parameter.
    """
    stability_results = np.load(params['fname_stability_results'],
                                allow_pickle=True).item()

    fig = plt.figure(
        figsize=(
            params['figwidth_1col'],
            params['figwidth_1col']))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    plt.subplots_adjust(
        bottom=0.19, top=0.95, left=0.15, right=0.93, wspace=1.2)

    ax = _plot_eigenvalues_wavenumber(gs[0, 0], stability_results)
    xshift = -0.6
    yshift = 0.02
    _add_label(ax, 'A', xshift=xshift, yshift=yshift)

    ax = _plot_eigenvalues_alpha(gs[0, 1], stability_results)
    _add_label(ax, 'B', xshift=xshift - 0.05, yshift=yshift)

    plt.savefig(params['figure_fname'] + '_eigenvalues.eps')


##########################################################################
# ======================================================================
# Helper functions for linear stability analyisis (linear interpolation)
# ======================================================================
def _solve_chareq_numerically_alpha(
        lambda_rate, k, alpha, network, tau_rate, W_rate):
    """
    Solves the full characteristic equation numerically.

    Parameters
    ----------
    lambda_rate: complex float
        Eigenvalue of rate model in 1/s.
    k : float
        Wave number in 1/m.
    alpha : float
        Interpolation parameter.
    network : nnmt.models.Network or child class instance
        Network instance.
    tau_rate : np.array
        Time constants of rate model in s.
    W_rate : np.array
        Weight matrix of rate model.

    Returns
    -------
    lamb :
        Numerically optimized eigenvalues as a function of the interpolation
        parameter.
    """
    def fsolve_complex(l_re_im):
        lam = complex(l_re_im[0], l_re_im[1])

        spatial_profile = spatial._ft_spatial_profile_boxcar(
            k=k, width=network.network_params['width'])

        eff_conn_spiking = linstab._linalg_max_eigenvalue(
            _effective_connectivity_spiking(lam, network) * spatial_profile)
        eff_conn_rate = linstab._linalg_max_eigenvalue(
            _effective_connectivity_rate(
                lam, tau_rate, W_rate) * spatial_profile)

        eff_conn_alpha = alpha * eff_conn_spiking + \
            (1. - alpha) * eff_conn_rate

        roots = eff_conn_alpha * np.exp(-lam * d) - 1.

        return [roots.real, roots.imag]

    delay_mean = network.network_params['D_mean']
    assert np.isscalar(delay_mean) or len(np.unique(delay_mean) == 1)
    d = np.unique(delay_mean)[0]

    lambda_guess_list = [np.real(lambda_rate), np.imag(lambda_rate)]
    l_opt = sopt.fsolve(fsolve_complex, lambda_guess_list)
    lamb = complex(l_opt[0], l_opt[1])
    return lamb


def _solve_lambda_of_alpha_integral(
        lambda_rate, k, alphas, network, tau_rate, W_rate):
    """
    Integrates the derivative of the eigenvalue wrt. interpolation parameters.

    Parameters
    ----------
    lambda_rate : complex float
        Eigenvalue of rate model in 1/s.
    k : float
        Wave number in 1/m.
    alphas : np.array of floats
        All interpolation parameters.
    network : nnmt.models.Network or child class instance
        Network instance.
    tau_rate : np.array
        Time constants of rate model in s.
    W_rate : np.array
        Weight matrix of rate model.

    Returns
    -------
    lambdas_of_alpha :
        Numerically integrated eigenvalues as a function of interpolation
        parameters.
    """
    assert alphas[0] == 0, 'First alpha must be 0!'
    lambda0_list = [lambda_rate.real, lambda_rate.imag]

    def derivative(lambda_list, alpha):
        lam = complex(lambda_list[0], lambda_list[1])
        deriv = _d_lambda_d_alpha(lam, alpha, k, network, tau_rate, W_rate)
        return [deriv.real, deriv.imag]

    llist = sint.odeint(func=derivative, y0=lambda0_list, t=alphas)

    lambdas_of_alpha = [complex(lam[0], lam[1]) for lam in llist]
    return lambdas_of_alpha


def _effective_connectivity_spiking(lam, network):
    """
    Computes the effective connectivity of the spiking model.

    Parameters
    ----------
    lam : complex float
        Eigenvalue in 1/s.
    network : nnmt.models.Network or child class instance
        Network instance.

    Returns
    -------
    eff_conn :
        Effective connectivity.
    """
    omega = complex(0, -lam)
    transfunc = mft.transfer_function(
        network=network, freqs=np.array([omega / (2. * np.pi)]))

    D = np.array([1])  # ignore delay distribution here
    eff_conn = mft._effective_connectivity(
        transfer_function=transfunc, D=D, J=network.network_params['J'],
        K=network.network_params['K'], tau_m=network.network_params['tau_m'])
    return eff_conn


def _effective_connectivity_rate(lam, tau_rate, W_rate):
    """
    Computes the effective connectivity of the rate model.

    Parameters
    ----------
    lam : complex float
        Eigenvalue in 1/s.
    network : nnmt.models.Network or child class instance
        Network instance.

    Returns
    -------
    eff_conn :
        Effective connectivity.
    """
    omega = complex(0, -lam)
    eff_conn = W_rate / (1. + 1j * omega * tau_rate)
    return eff_conn


def _d_lambda_d_alpha(lam, alpha, k, network, tau_rate, W_rate):
    """
    Computes the derivative of the eigenvalue wrt. the interpolation parameter.

    Parameters
    ----------
    lam : complex float
        Eigenvalue of rate model in 1/s.
    alpha : float
        Interpolation parameter.
    k : float
        Wave number in 1/m.
    network : nnmt.models.Network or child class instance
        Network instance.
    tau_rate : np.array
        Time constants of rate model in s.
    W_rate : np.array
        Weight matrix of rate model.

    Returns
    -------
    deriv :
        Derivative.
    """
    spatial_profile = spatial._ft_spatial_profile_boxcar(
        k=k, width=network.network_params['width'])

    eff_conn_spiking = linstab._linalg_max_eigenvalue(
        _effective_connectivity_spiking(lam, network) * spatial_profile)
    eff_conn_rate = linstab._linalg_max_eigenvalue(
        _effective_connectivity_rate(lam, tau_rate, W_rate) * spatial_profile)

    eff_conn_alpha = alpha * eff_conn_spiking + (1. - alpha) * eff_conn_rate

    d_eff_conn_spiking_d_lambda = linstab._linalg_max_eigenvalue(
        _d_eff_conn_spiking_d_lambda(lam, network) * spatial_profile)

    d_eff_conn_rate_d_lambda = linstab._linalg_max_eigenvalue(
        _d_eff_conn_rate_d_lambda(lam, tau_rate, W_rate) * spatial_profile)

    d_eff_conn_alpha_d_lambda = alpha * d_eff_conn_spiking_d_lambda + \
        (1. - alpha) * d_eff_conn_rate_d_lambda

    delay_mean = network.network_params['D_mean']
    assert np.isscalar(delay_mean) or len(np.unique(delay_mean) == 1)
    d = np.unique(delay_mean)[0]

    nominator = eff_conn_spiking - eff_conn_rate
    denominator = d_eff_conn_alpha_d_lambda - d * eff_conn_alpha

    deriv = - nominator / denominator
    return deriv


def _d_eff_conn_spiking_d_lambda(lam, network):
    """
    Computes the derivative of the effective connectivity of the spiking model.

    Parameters
    ----------
    l : complex float
        Eigenvalue of rate model in 1/s.
    network : nnmt.models.Network or child class instance
        Network instance.

    Returns
    -------
    deriv :
        Derivative.
    """
    def f(x):
        return _effective_connectivity_spiking(x, network)
    deriv = smisc.derivative(func=f, x0=lam, dx=1e-10)
    return deriv


def _d_eff_conn_rate_d_lambda(lam, tau_rate, W_rate):
    """
    Computes the derivative of the effective connectivity of the rate model.

    Parameters
    ----------
    l : complex float
        Eigenvalue of rate model in 1/s.
    tau_rate : np.array
        Time constants of rate model in s.
    W_rate : np.array
        Weight matrix of rate model.

    Returns
    -------
    deriv:
        Derivative.
    """
    lp = 1. / (1. + lam * tau_rate)
    deriv = -1. * W_rate * lp**2 * tau_rate
    return deriv


##########################################################################
# ====================================================
# Plot functions for network illustrations and results
# ====================================================
def _plot_network_sketch_sun(gs_glob):
    """
    Illustrates ring-like network structure.

    Parameters
    ----------
    gs_glob : GridSpec cell
        Global GridSpec cell to plot into.
    """
    ax = plt.subplot(gs_glob)

    circ = Circle(xy=(0, 0), radius=1, color='k', fill=False)
    ax.add_artist(circ)

    blue = params['colors']['ex_blue']
    red = params['colors']['inh_red']

    n_in = 10
    ex2in = 4

    step = 2 * np.pi / n_in

    offset = 0
    for i in np.arange(1 + ex2in):
        if i == 0:  # inh.
            marker = 'o'
            color = red
        else:  # exc.
            marker = '^'
            color = blue

        xy = []
        for i in np.arange(n_in):
            x = (1 + offset) * np.cos(i * step)
            y = (1 + offset) * np.sin(i * step)
            xy.append([x, y])
        xy = np.array(xy)

        ax.plot(xy[:, 0], xy[:, 1],
                marker=marker, markersize=mpl.rcParams['lines.markersize'],
                linestyle='', color=color)
        offset += 0.2

    ax.annotate('ring\n network', (0, 0), va='center', ha='center')

    lim = 2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')

    plt.axis('off')
    return ax


def _plot_spatial_profile(gs_glob):
    """
    Plots spatial conenctivity profile for an exc. and an inh. population.

    Parameters
    ----------
    gs_glob : GridSpec cell
        Global GridSpec cell to plot into.
    """
    def _get_p(rs, width):
        p = np.zeros(len(rs))
        # p is normalized to 1
        height = 1. / (2 * width)
        p[np.where(np.abs(rs) <= width)] = height
        return p

    gs = gs_glob.subgridspec(4, 1)

    ax = plt.subplot(gs[1:3])

    blue = params['colors']['ex_blue']
    red = params['colors']['inh_red']

    network = BasicNetwork(
        network_params='parameters/Senk2020_network_params.yaml',
        analysis_params='parameters/Senk2020_analysis_params.yaml')
    ewidth = network.network_params['width'][0] * \
        params['quantities']['displacement']['scale']
    iwidth = network.network_params['width'][1] * \
        params['quantities']['displacement']['scale']

    max_x = np.max([ewidth, iwidth])
    rs = np.arange(-1.5 * max_x, 1.5 * max_x, 1e-5)  # in mm

    ep = _get_p(rs, ewidth)
    ip = _get_p(rs, iwidth)
    ax.plot(rs, ep, blue)
    ax.plot(rs, ip, red)

    xstp = ewidth / 6.
    ax.annotate('E',
                [ewidth + xstp, 1. / (2. * ewidth)],
                color=blue,
                va='top', ha='left')

    ax.annotate('I',
                [iwidth + xstp, 1. / (2. * iwidth)],
                color=red,
                va='top', ha='left')

    ax.set_xlim(rs[0], rs[-1])
    ax.set_xlabel(params['quantities']['displacement']['label'])
    ax.get_yaxis().set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title('connection\n probability $p$')
    return ax


def _plot_mean_std_images(gs_glob, tf_scan_results):
    """
    Creates image plots for results from scan_fit_transfer_function().

    Parameters
    ----------
    gs_glob : GridSpec cell
        Global GridSpec cell to plot into.
    tf_scan_results : dict
       Loaded results from scan_fit_transfer_function().
    """
    gs = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=gs_glob, hspace=0.3, wspace=0)

    mus = params['mean_inputs_scan']  # first index
    sigmas = params['std_inputs_scan']  # second index
    mu_star = params['mean_std_inputs_stability'][0]
    sigma_star = params['mean_std_inputs_stability'][1]

    axes = []
    for k, key in enumerate(['nu_ext_exc', 'nu_ext_inh', 'firing_rates',
                             'tau_rate', 'W_rate', 'fit_error']):
        ax = plt.subplot(gs[k])
        axes.append(ax)
        img = ax.pcolormesh(
            np.transpose(
                tf_scan_results[key] * params['quantities'][key]['scale']))

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

        cb = plt.colorbar(img)
        cb.ax.tick_params(pad=0)
        cb.locator = ticker.MaxNLocator(nbins=4)
        cb.update_ticks()
        # star for mu and sigma used in this circuit (0.5 offset for
        # pcolormesh)
        xmu = np.max(ax.get_xticks() - 0.5) * (mu_star - np.min(mus)) \
            / (np.max(mus) - np.min(mus)) + 0.5
        ysigma = np.max(ax.get_yticks() - 0.5) \
            * (sigma_star - np.min(sigmas)) \
            / (np.max(sigmas) - np.min(sigmas)) + 0.5

        ax.plot(xmu, ysigma,
                marker='*', markerfacecolor='white', markeredgecolor='none',
                markersize=mpl.rcParams['lines.markersize'] * 2.5)
        ax.plot(xmu, ysigma,
                marker='*', markerfacecolor='k', markeredgecolor='none',
                markersize=mpl.rcParams['lines.markersize'] * 2.)

        ax.set_title(params['quantities'][key]['label'])

    return axes


def _plot_transfer_functions(gs_glob, tf_scan_results):
    """
    Plots transfer function and fit for selection of parameters.

    Uses results computed with scan_fit_transfer_function().

    Parameters
    ----------
    gs_glob : GridSpec cell
        Global GridSpec cell to plot into.
    tf_scan_results : dict
       Loaded results from scan_fit_transfer_function().
    """
    gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_glob, hspace=0)
    ax_amplitude = plt.subplot(gs[0])
    ax_phase = plt.subplot(gs[1])

    leg_handles, leg_labels = [[], []], []
    c = 0
    for i, mu in enumerate(params['mean_inputs_scan']):
        for j, sigma in enumerate(params['std_inputs_scan']):
            if [mu, sigma] in params['mean_std_inputs_tf'].tolist():
                cols = [params['colors'][x] for x in params['colors_tf'][c]]

                transfer_function = \
                    tf_scan_results['transfer_function'][i, j] \
                    * params['quantities']['transfer_function']['scale']

                transfer_function_fit = \
                    tf_scan_results['transfer_function_fit'][i, j] \
                    * params['quantities']['transfer_function']['scale']

                frequencies = tf_scan_results['frequencies'] \
                    * params['quantities']['frequencies']['scale']

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
                    [ax_amplitude,
                     ax_phase],
                    [params['quantities']['transfer_function_amplitude'][
                        'label'],
                     params['quantities']['transfer_function_phase'][
                         'label']]):

                    if any(frequencies > 0):
                        ax.set_xscale('log')
                    ax.set_ylabel(ylabel)
                    ax.set_xlim(frequencies[0], frequencies[-1])
                ax_amplitude.set_xticklabels([])

                leg_handles[0].append(Patch(facecolor=cols[0]))
                leg_handles[1].append(Patch(facecolor=cols[1]))
                leg_labels.append(f'({int(mu * 1e3)}, {int(sigma * 1e3)})')
                c += 1

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
        [],
        [],
        color='k',
        linestyle='none',
        marker='o',
        markersize=mpl.rcParams['lines.markersize'] *
        0.1,
        label='rate model (fit)')
    ax_amplitude.legend(handles=[spiking, rate])
    return ax_amplitude


def _plot_eigenvalues_wavenumber(gs_glob, stability_results):
    """
    Plots eigenvalues from rate model vs. wavenumbers.

    Uses results computed with linear_stability_analysis().

    Parameters
    ----------
    gs_glob : GridSpec cell
        Global GridSpec cell to plot into.
    stability_results : dict
        Loaded results from linear_stability_analysis().
    """
    gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_glob, hspace=0.1)

    ax_real = plt.subplot(gs[0])
    ax_imag = plt.subplot(gs[1])

    if (params['quantities']['eigenvalues_real']['scale'] ==
            params['quantities']['eigenvalues_imag']['scale']):
        scale_ev = params['quantities']['eigenvalues_real']['scale']

    branches = stability_results['branches']
    k_wavenumbers = stability_results['k_wavenumbers']
    eigenvalues = stability_results['eigenvalues']

    ks = k_wavenumbers * params['quantities']['k_wavenumbers']['scale']
    for i, branch_nr in enumerate(branches):
        ax_real.plot(ks, np.real(eigenvalues)[i] * scale_ev,
                     color=params['colors'][params['colors_br'][branch_nr]],
                     label=branch_nr)
        ax_real.set_ylabel(params['quantities']['eigenvalues_real']['label'],
                           labelpad=5.5)

        ax_imag.plot(ks, np.imag(eigenvalues)[i] * scale_ev,
                     color=params['colors'][params['colors_br'][branch_nr]])
        ax_imag.set_ylabel(params['quantities']['eigenvalues_imag']['label'],
                           labelpad=0)

    ax_real.set_title(params['quantities']['eigenvalues']['label'])
    ax_real.set_xticklabels([])
    # add whitespace via new lines to match axes of alpha plot
    ax_imag.set_xlabel(params['quantities']['k_wavenumbers']['label'] + '\n\n')

    # legend
    labels = ['0', '-1', '1', '-2', '2', '-3']  # ordered
    handles_old, labels_old = ax_real.get_legend_handles_labels()
    handles = []
    for lam in labels:
        for i, lo in enumerate(labels_old):
            if lam == lo:
                handles.append(handles_old[i])
    ax_real.legend(handles, labels, title='branch number', ncol=3,
                   columnspacing=0.1, loc='center', bbox_to_anchor=(0.55, 0.2))

    # find index where imag. of principle branch becomes 0 for xlims
    lambdas_imag = np.imag(eigenvalues)
    idx_b0 = np.where(np.array(branches) == 0)[0][0]
    # first index where imag. goes to 0
    idx_0 = np.where(np.array(lambdas_imag[idx_b0]) == 0)[0][0]
    offset = 5  # manual offset
    klim = ks[idx_0 - offset]

    for ax in [ax_real, ax_imag]:
        ax.axhline(0, linestyle='-', color='k',
                   linewidth=mpl.rcParams['lines.linewidth'] * 0.5)
        ax.set_xlim(ks[0], klim)

    # star for maximum real part
    for ax, fun in zip([ax_real, ax_imag],
                       [np.real, np.imag]):
        ax.plot(stability_results['k_eigenval_max']
                * params['quantities']['k_wavenumbers']['scale'],
                np.abs(fun(stability_results['eigenval_max']) * scale_ev),
                marker='*', markerfacecolor='k', markeredgecolor='none',
                markersize=mpl.rcParams['lines.markersize'] * 2.)
    return ax_real


def _plot_eigenvalues_alpha(gs_glob, stability_results):
    """
    Plots linear interpolation of eigenvalues.

    Uses results computed with linear_stability_analysis().

    Parameters
    ----------
    gs_glob : GridSpec cell
        Global GridSpec cell to plot into.
    stability_results : dict
        Loaded results from linear_stability_analysis().
    """
    gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_glob, hspace=0.1)

    ax_real = plt.subplot(gs[0])
    ax_imag = plt.subplot(gs[1])

    if (params['quantities']['eigenvalues_real']['scale'] ==
            params['quantities']['eigenvalues_imag']['scale']):
        scale_ev = params['quantities']['eigenvalues_real']['scale']

    branches = stability_results['branches']
    alphas = stability_results['alphas']
    lambdas_integral = stability_results['lambdas_integral']
    lambdas_chareq = stability_results['lambdas_chareq']

    xlim = [-0.1, 1.1]
    for i, branch_nr in enumerate(branches):
        for ax, fun, lab in zip([ax_real, ax_imag],
                                [np.real, np.imag],
                                ['real', 'imag']):
            ax.plot(alphas,
                    fun(lambdas_integral[i]) * scale_ev,
                    linestyle='',
                    color=params['colors'][params['colors_br'][branch_nr]],
                    marker='o',
                    markersize=mpl.rcParams['lines.markersize'] * 1.5,
                    markeredgecolor='none')
            ax.plot(alphas, fun(lambdas_chareq[i]) * scale_ev,
                    linestyle='-',
                    color=params['colors'][params['colors_br'][branch_nr]],
                    markeredgecolor='none')
            ax.set_ylabel(params['quantities']['eigenvalues_' + lab]['label'],
                          labelpad=0)

            # lambda = 0
            ax.plot(xlim, [0, 0], 'k-',
                    linewidth=mpl.rcParams['lines.linewidth'] * 0.5)
            ax.set_xlim(xlim[0], xlim[1])

            # star marker
            if branch_nr == 0:
                ax.plot(0,
                        np.abs(fun(stability_results['eigenval_max']))
                        * scale_ev,
                        marker='*',
                        markerfacecolor='k',
                        markeredgecolor='none',
                        markersize=mpl.rcParams['lines.markersize'] * 2.)

    xticks = [0, 0.5, 1]
    ax_real.set_title(params['quantities']['eigenvalues']['label'])
    ax_real.set_xticks(xticks)
    ax_real.set_xticklabels([])
    ax_imag.set_xlabel(r'interpolation parameter $\alpha$')
    ax_imag.set_xticks(xticks)
    ax_imag.set_xticklabels(['0.0\nrate\nmodel', '0.5', '1.0\nspiking\nmodel'])

    # legend for symbols
    integral = mpl.lines.Line2D([], [], color='k',
                                marker='o', linestyle='', label='integral')
    chareq = mpl.lines.Line2D([], [], color='k',
                              marker=None, linestyle='-', label='char. eq.')
    ax_real.legend(
        handles=[integral, chareq], bbox_to_anchor=(0.6, 0.7), loc='center')
    return ax_real


def _add_label(ax, label, xshift=0., yshift=0., scale_fs=1.):
    """
    Add label to plot panel given by axis.

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
# ===========================
# Execution of main functions
# ===========================
if __name__ == '__main__':

    scan_fit_transfer_function()

    linear_stability_analysis()

    figure_Senk2020_network_structure()

    figure_Senk2020_input_scan()

    figure_Senk2020_eigenvalues()
