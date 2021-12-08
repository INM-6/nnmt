"""
Spatial patterns
================

This example demonstrates the methods used in Figure 6 of :cite:t:`senk2020`.
A figure illustrating the network structure of the used model is set up in
:doc:`network_structure`.
The same model is used in the example :doc:`mapping_lif_rate`.

"""

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
plt.style.use('frontiers.mplstyle')
mpl.rcParams.update({'legend.fontsize': 'medium',  # old: 5.0 was too small
                     'axes.titlepad': 0.0,
                     'figure.constrained_layout.use': False})

#############################################################################
# First, we define parameters for data generation and plotting.

params = {
    # mean and standard deviation of input used for stability analysis (in V)
    'mean_std_inputs_stability': np.array([10., 10.]) * 1e-3,

    # labels and corresponding scaling parameters for plotted quantities
    'quantities': {
        'k_wavenumbers': {
            'label': 'wavenumber $k$ (1/mm)',
            'scale': 1e-3},
        'eigenvalues': {
            'label': r'eigenvalue $\lambda$'},
        'eigenvalues_real': {
            'label': r'Re[$\lambda$] (1000/s)',
            'scale': 1e-3},
        'eigenvalues_imag': {
            'label': r'Im[$\lambda$] (1000/s)',
            'scale': 1e-3}},

    # figure width in inch
    'figwidth_1col': 85. / 25.4,

    # colors for branches of Lambert W function
    'colors_br': {0: '#994455',  # dark red
                  -1: '#EE99AA',  # light red
                  1: '#004488',  # dark blue
                  -2: '#6699CC',  # light blue
                  2: '#997700',  # dark yellow
                  -3: '#EECC66'}}  # light yellow

##########################################################################
# ================
# Helper functions
# ================
# Here, we define a number of helper functions which are currently considered
# too specific for a global integration into NNMT.
# These functions are concerned with solving the characteristic equation of the
# spiking and rate models and interpolating between them.


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

        eff_conn_spiking = _linalg_max_eigenvalue(
            _effective_connectivity_spiking(lam, network) * spatial_profile)
        eff_conn_rate = _linalg_max_eigenvalue(
            _effective_connectivity_rate(
                lam, tau_rate, W_rate) * spatial_profile)

        eff_conn_alpha = (alpha * eff_conn_spiking
                          + (1. - alpha) * eff_conn_rate)

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

    eff_conn_spiking = _linalg_max_eigenvalue(
        _effective_connectivity_spiking(lam, network) * spatial_profile)
    eff_conn_rate = _linalg_max_eigenvalue(
        _effective_connectivity_rate(lam, tau_rate, W_rate) * spatial_profile)

    eff_conn_alpha = alpha * eff_conn_spiking + (1. - alpha) * eff_conn_rate

    d_eff_conn_spiking_d_lambda = _linalg_max_eigenvalue(
        _d_eff_conn_spiking_d_lambda(lam, network) * spatial_profile)

    d_eff_conn_rate_d_lambda = _linalg_max_eigenvalue(
        _d_eff_conn_rate_d_lambda(lam, tau_rate, W_rate) * spatial_profile)

    d_eff_conn_alpha_d_lambda = (alpha * d_eff_conn_spiking_d_lambda
                                 + (1. - alpha) * d_eff_conn_rate_d_lambda)

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


def _linalg_max_eigenvalue(matrix):
    """
    Computes the eigenvalue with the largest absolute value of a given matrix.

    Note that this a general matrix operation and the eigenvalue should not be
    confused with lambda, the temporal eigenvalue of a characteristic
    equation.

    Parameters
    ----------
    matrix : np.array
        Matrix to calculate eigenvalues from.

    Returns
    -------
    max_eigval : float
        Maximum eigenvalue.
    """
    eigvals = np.linalg.eigvals(matrix)
    max_eigval = eigvals[np.argmax(np.abs(eigvals))]
    return max_eigval

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

print('Instantiating network model.')

network = BasicNetwork(
    network_params='Senk2020_network_params.yaml',
    analysis_params='Senk2020_analysis_params.yaml')

##########################################################################
# The working point is set with a given mean and standard deviation of the
# input.
# The excitatory and inhibitory external firing rates required to preserve this
# working point are computed and the network parameters adjusted.
# Then, we calculate the LIF transfer function and fit it with the one of a
# low-pass filter using a least-squares fit.

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

##########################################################################
# The first loop for generating data iterates over branches of the Lambert W
# function and wave numbers of the spatial connectivity profile.
# With the rate model and the parameters obtained by fitting the LIF transfer
# function, we can calculate an anlytical solution for the eigenvalues solving
# the characteristic equation.

print('Solving characteristic equation for rate model analytically.')

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

##########################################################################
# Then, we perform a linear interpolation of eigenvalues from the rate model
# (analytical) to the spiking model (LIF, only numerical) by two methods and
# loop here over branch numbers:
#  1. solving the full characteristic equation numerically and
#  2. integrating the derivative of the eigenvalue with respect to the
#     interpolation parameter.

print('Linear interpolation between rate and spiking models.')

alphas = np.linspace(0, 1, 5)
lambdas_integral = np.zeros((len(branches), (len(alphas))), dtype=complex)
lambdas_chareq = np.zeros((len(branches), len(alphas)), dtype=complex)
for i, branch_nr in enumerate(branches):
    print(f'    branch number = {branch_nr}')

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

##########################################################################
# All results are stored in ``stability_results``.

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

##########################################################################
# ========
# Plotting
# ========
# We generate a figure with two panels
# The figure spans one column.

print('Plotting.')

fig = plt.figure(figsize=(params['figwidth_1col'], params['figwidth_1col']))
gs = gridspec.GridSpec(1, 2, figure=fig)
plt.subplots_adjust(bottom=0.19, top=0.95, left=0.15, right=0.93, wspace=1.2)

##########################################################################
# To panel A, we plot eigenvalues from the rate model vs. wavenumbers.

gs_rt = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs[0, 0], hspace=0.1)

ax_real = plt.subplot(gs_rt[0])
_add_label(ax_real, 'A', xshift=-0.6, yshift=0.02)
ax_imag = plt.subplot(gs_rt[1])

if (params['quantities']['eigenvalues_real']['scale'] ==
        params['quantities']['eigenvalues_imag']['scale']):
    scale_ev = params['quantities']['eigenvalues_real']['scale']

branches = stability_results['branches']
k_wavenumbers = stability_results['k_wavenumbers']
eigenvalues = stability_results['eigenvalues']

ks = k_wavenumbers * params['quantities']['k_wavenumbers']['scale']
for i, branch_nr in enumerate(branches):
    ax_real.plot(ks, np.real(eigenvalues)[i] * scale_ev,
                 color=params['colors_br'][branch_nr],
                 label=branch_nr)
    ax_real.set_ylabel(params['quantities']['eigenvalues_real']['label'],
                       labelpad=5.5)

    ax_imag.plot(ks, np.imag(eigenvalues)[i] * scale_ev,
                 color=params['colors_br'][branch_nr])
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

##########################################################################
# To panel B, we plot the linear interpolation of eigenvalues.

gs_int = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs[0, 1], hspace=0.1)

ax_real = plt.subplot(gs_int[0])
_add_label(ax_real, 'B', xshift=-0.65, yshift=0.02)
ax_imag = plt.subplot(gs_int[1])

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
                color=params['colors_br'][branch_nr],
                marker='o',
                markersize=mpl.rcParams['lines.markersize'] * 1.5,
                markeredgecolor='none')
        ax.plot(alphas, fun(lambdas_chareq[i]) * scale_ev,
                linestyle='-',
                color=params['colors_br'][branch_nr],
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

##########################################################################
# The final figure is saved to file.

plt.savefig('spatial_patterns.eps')
