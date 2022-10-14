"""
Collection of functions used by both `lif.delta` and `lif.exp`.

Static Quantities
*****************

.. autosummary::
    :toctree: _toctree/lif/

    _input_calc
    mean_input
    _mean_input
    std_input
    _std_input
    _fit_transfer_function

"""

import numpy as np
import scipy.optimize as sopt

from .. import ureg


def _input_calc(network, prefix, input_func):
    '''
    Helper function for input related calculations.

    Checks the requirements for calculating input related quantities and calls
    the respective input function.

    Parameters
    ----------
    network : nnmt.create.Network object
        The network for which the calculation should be done.
    prefix : str
        The prefix used to store the results (e.g. 'lif.delta.').
    input_func : function
        The function that should be calculated (either `_mean_input` or
        `_std_input`).
    '''
    try:
        rates = (
            network.results[prefix + 'firing_rates'].to_base_units().magnitude)
    except KeyError as quantity:
        raise RuntimeError(f'You first need to calculate the {quantity}.')
    list_of_params = ['K', 'J', 'tau_m', 'nu_ext', 'K_ext', 'J_ext']
    try:
        params = {key: network.network_params[key] for key in list_of_params}
    except KeyError as param:
        raise RuntimeError(f'You are missing {param} for this calculation.')

    return input_func(rates, **params) * ureg.V


def _mean_input(nu, J, K, tau_m, J_ext, K_ext, nu_ext, I_ext=None, C=None):
    """
    Calc mean input for lif neurons in fixed in-degree connectivity network.

    Following Eq. 3.4 in :cite:t:`fourcaud2002`.

    Parameters
    ----------
    nu : np.array
        Firing rates of populations in Hz.
    J : np.array
        Weight matrix in V.
    K : np.array
        In-degree matrix.
    tau_m : [float | 1d array]
        Membrane time constant of post-synatic neuron in s.
    J_ext : np.array
        External weight matrix in V.
    K_ext : np.array
        Numbers of external input neurons to each population.
    nu_ext : 1d array
        Firing rates of external populations in Hz.
    I_ext : [float | np.array], optional
        External d.c. input in A, requires membrane capacitance as well.
    C : [float | np.array], optional
        Membrane capacitance in F, required if external input is given.

    Returns
    -------
    np.array
        Array of mean inputs to each population in V.
    """
    # contribution from within the network
    m = np.dot(K * J, nu)
    # contribution from external sources
    m += np.dot(K_ext * J_ext, nu_ext)
    # contribution of external input current
    if I_ext and C:
        m += I_ext / C
    m *= tau_m
    return m


def _std_input(nu, J, K, tau_m, J_ext, K_ext, nu_ext):
    """
    Calc std of input for lif neurons in fixed in-degree connectivity network.

    Following Eq. 3.4 in :cite:t:`fourcaud2002`.

    Parameters
    ----------
    nu : np.array
        Firing rates of populations in Hz.
    J : np.array
        Weight matrix in V.
    K : np.array
        In-degree matrix.
    tau_m : [float | 1d array]
        Membrane time constant of post-synatic neuron in s.
    J_ext : np.array
        External weight matrix in V.
    K_ext : np.array
        Numbers of external input neurons to each population.
    nu_ext : 1d array
        Firing rates of external populations in Hz.

    Returns
    -------
    np.array
        Array of standard deviation of inputs to each population in V.
    """
    # contribution from within the network to variance
    var = tau_m * np.dot(K * J**2, nu)
    # contribution from external sources to variance
    var += tau_m * np.dot(K_ext * J_ext**2, nu_ext)
    # standard deviation is square root of variance
    return np.sqrt(var)


def _fit_transfer_function(transfunc, omegas):
    """
    Fits the transfer function (tf) of a low-pass filter to the passed tf.

    A least-squares fit is used for the fitting procedure.

    For details refer to
    :cite:t:`senk2020`, Sec. F 'Comparison of neural-field and spiking models'.

    Parameters
    ----------
    transfer_function : np.array
        Transfer functions for each population with the following shape:
        (number of freqencies, number of populations).
    omegas : [float | np.ndarray]
        Input frequencies to population in Hz.

    Returns
    -------
    transfer_function_fit : np.array
        Fit of transfer functions in Hertz/volt for each population with the
        following shape: (number of freqencies, number of populations).
    tau_rate : np.array
        Fitted time constant of low-pass filter for each population in s.
    h0 : np.array
        Fitted gain of low-pass filter for each population in Hertz/volt.
    fit_error : float
        Combined fit error.
    """
    def func(omega, tau, h0):
        return h0 / (1. + 1j * omega * tau)

    # absolute value for fitting
    def func_abs(omega, tau, h0):
        return np.abs(func(omega, tau, h0))

    transfunc_fit = np.zeros(np.shape(transfunc), dtype=np.complex_)
    dim = np.shape(transfunc)[1]
    tau_rate = np.zeros(dim)
    h0 = np.zeros(dim)
    fit_error = np.zeros(dim)

    for i in np.arange(dim):
        # fit low-pass filter transfer function (func) to LIF transfer function
        # (transfunc) to obtain parameters of rate model with fit errors
        fitParams, fitCovariances = sopt.curve_fit(
            func_abs, omegas, np.abs(transfunc[:, i]))

        tau_rate[i] = fitParams[0]
        h0[i] = fitParams[1]
        transfunc_fit[:, i] = func(omegas, tau_rate[i], h0[i])

        # adjust sign of imaginary part (just check sign of last value)
        sign_imag = 1 if (transfunc[-1, i].imag > 0) else -1
        sign_imag_fit = 1 if (transfunc_fit[-1, i].imag > 0) else -1
        if sign_imag != sign_imag_fit:
            transfunc_fit[:, i].imag *= -1
            tau_rate[i] *= -1

        # standard deviation
        fit_errs = np.sqrt(np.diag(fitCovariances))
        # relative error
        err_tau = fit_errs[0] / tau_rate[i]
        err_h0 = fit_errs[1] / h0[i]

        # combined error
        fit_error[i] = np.sqrt(err_tau**2 + err_h0**2)

    return transfunc_fit, tau_rate, h0, fit_error
