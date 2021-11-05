"""
Collection of functions used by both `lif.delta` and `lif.exp`.

Static Quantities
*****************

.. autosummary::
    :toctree: _toctree/lif/

    _firing_rate_integration
    _input_calc
    mean_input
    _mean_input
    std_input
    _std_input
    _fit_transfer_function

"""

from functools import partial
import numpy as np
import scipy.integrate as sint
import scipy.optimize as sopt

from .. import ureg


def _firing_rate_integration(firing_rate_func, firing_rate_params,
                             input_params, nu_0=None, fixpoint_method='ODE',
                             eps_tol=1e-7, t_max_ODE=1000, maxiter_ODE=1000):
    """
    Solves the self-consistent eqs for firing rates, mean and std of input.

    Parameters
    ----------
    firing_rate_func : func
        Function to be integrated.
    firing_rates_params : dict
        Parameters passed to firing_rates_func
    input_params : dict
        Parameters passed to functions calculating mean and std of input.
    nu_0 : [None | np.ndarray]
        Initial guess for fixed point integration. If `None` the initial guess
        is 0 for all populations. Default is `None`.
    fixpoint_method : str
        Method used for finding the fixed point. Currently, the following
        method are implemented: `ODE`, `LSQTSQ`. ODE is a very good choice,
        which finds stable fixed points even if the initial guess is far from
        the fixed point. LSQTSQ also finds unstable fixed points but needs a
        good initial guess. Default is `ODE`.

        ODE :
            Solves the initial value problem
              dnu / ds = - nu + firing_rate_func(nu)
            with initial value `nu_0` on the interval [0, t_max_ODE].
            The final value at `t_max_ODE` is used as a new initial value
            and the initial value problem is solved again. This procedure
            is iterated until the criterion for a self-consistent solution
              max( abs(nu[t_max_ODE-1] - nu[t_max_ODE]) ) < eps_tol
            is fulfilled. Raises an error if this does not happen within
            `maxiter_ODE` iterations.

        LSQTSQ :
            Determines the minimum of
              (nu - firing_rate_func(nu))^2
            using least squares. Raises an error if the solution is a local
            minimum with mean squared differnce above eps_tol.
    eps_tol : float
        Maximal incremental stepsize at which to stop the iteration procedure.
        Default is 1e-7.
    t_max_ODE : int
        Determines the interval [0, t_max_ODE] on which the initial value
        problem for the method `ODE` is solved in a single iteration.
        Default is 1000.
    maxiter_ODE : int
        Determines the maximum number of iterations of the initial value
        problem for the method `ODE`. Default is 1000.
    """

    dimension = input_params['K'].shape[0]

    def get_rate_difference(_, nu, rate_func):
        """
        Calculate difference between new iteration step and previous one.
        """
        # new mean
        mu = _mean_input(nu=nu, **input_params)
        # new std
        sigma = _std_input(nu=nu, **input_params)
        new_nu = rate_func(mu=mu, sigma=sigma, **firing_rate_params)

        return -nu + new_nu

    get_rate_difference = partial(get_rate_difference,
                                  rate_func=firing_rate_func)

    if nu_0 is None:
        nu_0 = np.zeros(int(dimension))

    if fixpoint_method == 'ODE':
        # do iteration procedure, until stationary firing rates are found
        for _ in range(maxiter_ODE):
            sol = sint.solve_ivp(get_rate_difference, [0, t_max_ODE], nu_0,
                                 t_eval=[t_max_ODE - 1, t_max_ODE],
                                 method='LSODA')
            assert sol.success is True
            eps = max(np.abs(sol.y[:, 1] - sol.y[:, 0]))
            if eps < eps_tol:
                return sol.y[:, 1]
            else:
                nu_0 = sol.y[:, 1]
        msg = f'Iteration failed to converge after {maxiter_ODE} steps. '
        msg += f'Last maximum difference {eps:e}, desired {eps_tol:e}.'
        raise RuntimeError(msg)
    elif fixpoint_method == 'LSTSQ':
        # search roots using least squares
        get_rate_difference = partial(get_rate_difference, None)
        res = sopt.least_squares(get_rate_difference, nu_0, bounds=(0, np.inf))
        if res.cost/dimension < eps_tol:
            return res.x
        else:
            msg = 'Least squares converged in a local minimum. '
            msg += f'Mean squared differences: {res.cost/dimension}.'
            raise RuntimeError(msg)
    else:
        msg = f"The method '{fixpoint_method}' to determine the self-"
        msg += "consistent fixpoint is not implemented."
        raise NotImplementedError(msg)


def mean_input(network, prefix):
    '''
    Calcs mean input for `network` and stores results using `prefix`.

    See :func:`nnmt.lif._general._mean_input` for full documentation.

    Parameters
    ----------
    network : Network object
        The network model for which the mean input is to be calculated. Needs
        to contain the parameters defined in
        :func:`nnmt.lif._general._mean_input`.
    prefix : str
        The prefix used to store the result (e.g. 'lif.delta.').

    Returns
    -------
    np.array
        Array of mean inputs to each population in V.

    See Also
    --------
    nnmt.lif._general._mean_input : For full documentation of network
                                    parameters.
    '''
    return _input_calc(network, prefix, _mean_input)


def std_input(network, prefix):
    '''
    Calcs std of input for `network` and stores results using `prefix`.

    See :func:`nnmt.lif._general._std_input` for full documentation.

    Parameters
    ----------
    network : Network object
        The network model for which the mean input is to be calculated. Needs
        to contain the parameters defined in
        :func:`nnmt.lif._general._std_input`.
    prefix : str
        The prefix used to store the result (e.g. 'lif.delta.').

    Returns
    -------
    np.array
        Array of standard deviation of inputs to each population in V.

    See Also
    --------
    nnmt.lif._general._std_input : For full documentation of network
                                   parameters.
    '''
    return _input_calc(network, prefix, _std_input)


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


def _mean_input(nu, J, K, tau_m, J_ext, K_ext, nu_ext):
    """
    Calc mean input for lif neurons in fixed in-degree connectivity network.

    Parameters
    ----------
    nu : np.array
        Firing rates of populations in Hz.
    J : np.array
        Weight matrix in V.
    K : np.array
        In-degree matrix.
    tau_m : [float | 1d array]
        Membrane time constant in s.
    J_ext : np.array
        External weight matrix in V.
    K_ext : np.array
        Numbers of external input neurons to each population.
    nu_ext : 1d array
        Firing rates of external populations in Hz.

    Returns
    -------
    np.array
        Array of mean inputs to each population in V.
    """
    # contribution from within the network
    m0 = np.dot(K * J, tau_m * nu)
    # contribution from external sources
    m_ext = tau_m * np.dot(K_ext * J_ext, nu_ext)
    # add them up
    m = m0 + m_ext
    return m


def _std_input(nu, J, K, tau_m, J_ext, K_ext, nu_ext):
    """
    Calc std of input for lif neurons in fixed in-degree connectivity network.

    Parameters
    ----------
    nu : np.array
        Firing rates of populations in Hz.
    J : np.array
        Weight matrix in V.
    K : np.array
        In-degree matrix.
    tau_m : [float | 1d array]
        Membrane time constant in s.
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
    var0 = np.dot(K * J**2, tau_m * nu)
    # contribution from external sources to variance
    var_ext = tau_m * np.dot(K_ext * J_ext**2, nu_ext)
    # add them up
    var = var0 + var_ext
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
        Fitted time constant for each population in s.
    h0 : ?
        ?
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
