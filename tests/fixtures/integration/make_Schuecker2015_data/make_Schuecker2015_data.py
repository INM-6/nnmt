# -*- coding: utf-8 -*-
"""
This script reproduces the data for the Figure 4 of the following
publication:

Schuecker, J., Diesmann, M. & Helias, M.
Modulated escape from a metastable state driven by colored noise.
Phys. Rev. E - Stat. Nonlinear, Soft Matter Phys. 92, 1–11 (2015).
"""

import matplotlib.pyplot as plt
import mpmath
import numpy as np
import scipy.special
import h5py_wrapper.wrapper as h5

import fortran_functions
import siegert

from collections import defaultdict


fix_path = 'integration/data/'


# Here we define the variants of the Phi1 function and change
# the function pointer of the actual Phi1 to the desired one

# in the original code, mpmath.pcfu() was used to compute the
# parabolic cylinder functions
def Phi1_mpmath_pcfu(z, x):
    '''mpmath implementation has revered sign in x'''
    try:
        value = np.exp(0.25 * x**2) * complex(mpmath.pcfu(z, -x))
        # value_mpc = U_Kummer_fortran(z,x)
        # value_mpc = U_Kummer_mpmath(z,x)
        # value = complex(value_mpc.real,value_mpc.imag)
        return value

    except mpmath.libmp.libhyper.NoConvergence:
        return np.NaN


# alternative computation of the parabolic cylinder function using
# a faster, but numerically unstable fortran implementation
def Phi1_U_Kummer_fortran(z, x):
    '''mpmath implementation has revered sign in x'''
    try:
        value_mpc = U_Kummer_fortran(z, x)
        value = complex(value_mpc.real, value_mpc.imag)
        return value

    except mpmath.libmp.libhyper.NoConvergence:
        return np.NaN
    

# alternative computation of the parabolic cylinder function using
# the more general hypergeometric function of first kind of mpmath
def Phi1_U_Kummer_mpmath(z, x):
    '''mpmath implementation has revered sign in x'''
    try:
        value_mpc = U_Kummer_mpmath(z, x)
        value = complex(value_mpc.real, value_mpc.imag)
        return value

    except mpmath.libmp.libhyper.NoConvergence:
        return np.NaN


def U_Kummer_fortran(a, x):
    factor = np.sqrt(np.pi) * 2**(-0.25 - 1 / 2. * a)
    factor_2 = np.sqrt(np.pi) * 2**(0.25 - 1 / 2. * a) * x
    first_term = factor * fortran_functions.kummers_function(
        0.5 * a + 0.25, 0.5, 0.5 * x**2) / mpmath.gamma(0.75 + 0.5 * a)
    second_term = factor_2 * fortran_functions.kummers_function(
        0.5 * a + 0.75, 1.5, 0.5 * x**2) / mpmath.gamma(0.25 + 0.5 * a)
    return first_term + second_term


def U_Kummer_mpmath(a, x):
    factor = np.sqrt(np.pi) * 2**(-0.25 - 1 / 2. * a)
    factor_2 = np.sqrt(np.pi) * 2**(0.25 - 1 / 2. * a) * x
    first_term = factor * mpmath.hyp1f1(
        0.5 * a + 0.25, 0.5, 0.5 * x**2) / mpmath.gamma(0.75 + 0.5 * a)
    second_term = factor_2 * mpmath.hyp1f1(
        0.5 * a + 0.75, 1.5, 0.5 * x**2) / mpmath.gamma(0.25 + 0.5 * a)
    return first_term + second_term


def d_Phi_DLMF_12_8_9(z, x):
    '''
    we use DLMF_12_8_9, but with negative sign, because DLMF use U
    to diverge at minus infinity (see. http://dlmf.nist.gov/12.3)
    '''
    return (1. / 2. + z) * Phi1(z + 1, x)


def transfer_FP_algebra_j1(omega, tau, tau_r, nu0, V_t, V_r, mu, sigma):
    if omega == 0.:
        return siegert.d_nu_d_mu(tau, tau_r, V_t, V_r, mu, sigma)
    else:
        x_t = np.sqrt(2.) * (V_t - mu) / sigma
        x_r = np.sqrt(2.) * (V_r - mu) / sigma
        z = complex(-0.5, complex(omega * tau))

        return np.sqrt(2.) / sigma * nu0 / (
            1. + complex(0., complex(omega * tau))) * (
                d_Phi_DLMF_12_8_9(z, x_r)
                - d_Phi_DLMF_12_8_9(z, x_t)) / (Phi1(z, x_r) - Phi1(z, x_t))


def transfer_FP_algebra_j1_shift(omega, tau, tau_s, tau_r, nu0, V_t, V_r, mu,
                                 sigma):

    alpha = np.sqrt(2) * abs(scipy.special.zetac(0.5) + 1)

    # effective threshold
    V_th1 = V_t + sigma * alpha / 2. * np.sqrt(tau_s / tau)

    # effective reset
    V_r1 = V_r + sigma * alpha / 2. * np.sqrt(tau_s / tau)

    return transfer_FP_algebra_j1(omega, tau, tau_r, nu0, V_th1, V_r1, mu,
                                  sigma)


def plot_PRE_Schuecker_Fig4(frequencies, sigma_1, mean_input_1,
                            sigma_2, mean_input_2):

    results_dict = defaultdict(str)

    results_dict['sigma'] = defaultdict(dict)

    for index in [1, 2]:
        sigma = eval('sigma_' + str(index))
        results_dict['sigma'][sigma]['mu'] = defaultdict(dict)
        for idx, mu in enumerate(eval('mean_input_' + str(index))):

            # Stationary firing rates for delta shaped PSCs.
            nu_0 = siegert.nu_0(tau_m, tau_r, theta, V_reset, mu, sigma)

            # Stationary firing rates for filtered synapses (via Taylor)
            nu0_fb = siegert.nu0_fb(
                tau_m, tau_s, tau_r, theta, V_reset, mu, sigma)

            # Stationary firing rates for exp PSCs. (via shift)
            nu0_fb433 = siegert.nu0_fb433(
                tau_m, tau_s, tau_r, theta, V_reset, mu, sigma)

            # colored noise zero-frequency limit of transfer function
            # colored noise
            transfer_function_zero_freq = siegert.d_nu_d_mu_fb433(
                tau_m, tau_s, tau_r, theta, V_reset, mu, sigma)

            transfer_function = [
                transfer_FP_algebra_j1_shift(
                    (2. * np.pi) * f, tau_m, tau_s, tau_r, nu0_fb, theta,
                    V_reset, mu, sigma)
                for f in frequencies
                ]

            results_dict['sigma'][sigma]['mu'][mu] = {
                'frequencies': frequencies,
                'absolute_value': np.abs(transfer_function),
                'phase': np.angle(transfer_function) / 2 / np.pi * 360,
                'zero_freq': transfer_function_zero_freq,
                'nu_0': nu_0,
                'nu0_fb': nu0_fb,
                'nu0_fb433': nu0_fb433}

            colors = ['black', 'grey']
            lw = 4
            markersize_cross = 4
            if sigma == 4.0:
                ls = '-'
            else:
                ls = '--'

            axA.semilogx(frequencies,
                         np.abs(transfer_function),
                         color=colors[idx],
                         linestyle=ls,
                         linewidth=lw,
                         label=r'$\nu=$ {} Hz'.format(nu0_fb))
            axB.semilogx(frequencies,
                         np.angle(transfer_function) / 2 / np.pi * 360,
                         color=colors[idx],
                         linestyle=ls,
                         linewidth=lw,
                         label=r'$\mu=$ ' + str(mu))
            axA.semilogx(zero_freq,
                         transfer_function_zero_freq,
                         '+',
                         color=colors[idx],
                         markersize=markersize_cross)

    axA.set_xlabel(r'frequency $\omega/2\pi\quad(1/\mathrm{s})$')
    axA.set_ylabel(r'$|\frac{n(\omega)\nu}{\epsilon\mu}|\quad(\mathrm{s}\,'
                   '\mathrm{mV})^{-1}$', labelpad=0)

    axB.set_xlabel(r'frequency $\omega/2\pi\quad(1/\mathrm{s})$')
    axB.set_ylabel(r'$-\angle n(\omega)\quad(^{\circ})$', labelpad=2)

    axA.legend()
    axB.legend()

    return results_dict


if __name__ == '__main__':

    tau_m = 20 * 1e-3  # membrane
    tau_s = 0.5 * 1e-3  # synapse
    tau_r = 0.0
    theta = 20.0  # threshold potential
    V_reset = 15.0  # reset potential

    # The following parameters are used in the function
    # sigma = 4.0, $\nu = 10, 30$ Hz
    mean_input_1 = np.array([16.42, 19.66])
    sigma_1 = 4.0

    # sigma = 1.5, $\nu = 10, 30$ Hz
    mean_input_2 = np.array([18.94, 20.96])
    sigma_2 = 1.5

    frequencies = np.logspace(-1, 2.8, num=500)
    zero_freq = 0.06

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 10))
    Phi1 = Phi1_mpmath_pcfu
    results_dict = plot_PRE_Schuecker_Fig4(frequencies, sigma_1, mean_input_1,
                                           sigma_2, mean_input_2)
    # fig.savefig(fix_path + 'PRE_Schuecker_Fig4.pdf')

    # save output
    h5.save(fix_path + 'Schuecker2015_data.h5', results_dict,
            overwrite_dataset=True)

    # Phi1 = Phi1_U_Kummer_fortran
    # plot_PRE_Schuecker_Fig4(['red', 'orange'], lw=2, markersize_cross=4)
    # axA.set_ylim(0, 15)
    # axB.set_ylim(-50, 10)
    # fig.savefig(fix_path + 'PRE_Schuecker_Fig4_Kummer_fortran.png')
    #
    # Phi1 = Phi1_U_Kummer_mpmath
    # plot_PRE_Schuecker_Fig4(['blue', 'green'], lw=2, markersize_cross=4)
    # axA.set_ylim(0, 15)
    # axB.set_ylim(-50, 10)
    # fig.savefig(fix_path + 'PRE_Schuecker_Fig4_Kummer_python.png')
