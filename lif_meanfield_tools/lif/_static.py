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
    nu_0 : None or np.ndarray
        Initial guess for fixed point integration. If `None` the initial guess
        is 0 for all populations. Default is `None`.
    fixpoint_method : str
        Method used for finding the fixed point. Currently, the following
        method are implemented: `ODE`, `LSQTSQ`. ODE is a very good choice,
        which finds stable fixed points even if the initial guess is far from
        the fixed point. LSQTSQ also finds unstable fixed points but needs a
        good initial guess. Default is `ODE`.

        ODE:
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
    Calc mean inputs to populations as function of firing rates of populations.

    See `delta.static.mean_input` or `exp.static.mean_input` for full
    documentation.
    '''
    return _input_calc(network, prefix, _mean_input)


def std_input(network, prefix):
    '''
    Calc std of inputs to populations as function of firing rates.

    See `delta.static.std_input` or `exp.static.std_input` for full
    documentation.
    '''
    return _input_calc(network, prefix, _std_input)


def _input_calc(network, prefix, input_func):
    '''
    Helper function for input related calculations.

    Checks the requirements for calculating input related quantities and calls
    the respective input function.

    Parameters:
    -----------
    network: lif_meanfield_tools.create.Network object
        The network for which the calculation should be done.
    prefix: str
        The prefix used in the to store the firing rates (e.g. 'lif.delta.').
    input_func: function
        The function that should be calculated (either mean or std).
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
    """ Compute mean input without quantities. """
    # contribution from within the network
    m0 = np.dot(K * J, tau_m * nu)
    # contribution from external sources
    tau_m = np.atleast_1d(tau_m)
    m_ext = np.dot(tau_m[np.newaxis].T * K_ext * J_ext, nu_ext)[0]
    # add them up
    m = m0 + m_ext
    return m


def _std_input(nu, J, K, tau_m, J_ext, K_ext, nu_ext):
    """ Compute standard deviation of input without quantities. """
    # contribution from within the network to variance
    var0 = np.dot(K * J**2, tau_m * nu)
    # contribution from external sources to variance
    tau_m = np.atleast_1d(tau_m)
    var_ext = np.dot(tau_m[np.newaxis].T * K_ext * J_ext**2, nu_ext)[0]
    # add them up
    var = var0 + var_ext
    # standard deviation is square root of variance
    return np.sqrt(var)
