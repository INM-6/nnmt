"""
Collection of general functions used to solve mean-field equations.

.. autosummary::
    :toctree: _toctree/lif/

    _firing_rate_integration

"""

from functools import partial
import numpy as np
import scipy.integrate as sint
import scipy.optimize as sopt


def _firing_rate_integration(firing_rate_func, firing_rate_params,
                             input_dict, nu_0=None,
                             fixpoint_method='ODE', eps_tol=1e-7,
                             t_max_ODE=1000, maxiter_ODE=1000):
    """
    Solves the self-consistent equations for firing rates.

    Parameters
    ----------
    firing_rate_func : func
        Function to be integrated.
    firing_rates_params : dict
        Parameters passed to firing_rates_func
    input_dict : dict
        Dictionary specifying the functions that need to be computed in each
        iteration step to get the input for the firing rate function. It should
        specify the names of the respective `firing_rate_func` arguments as
        keys and provide a dictionary for each input including the function
        using the key 'func' and the parameters using the key 'params':

        ``{'arg1': {'func': func1, 'params': params_dict1}, ...}``

        All input functions are assumed to take the firing rate as their first
        argument.
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

        LSTSQ :
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

    def get_rate_difference(_, nu, rate_func):
        """
        Calculate difference between new iteration step and previous one.
        """
        # new inputs
        inputs = {}

        for key, input in input_dict.items():
            inputs[key] = input['func'](nu, **input['params'])

        # new rate
        new_nu = rate_func(**{**firing_rate_params, **inputs})

        return -nu + new_nu

    get_rate_difference = partial(get_rate_difference,
                                  rate_func=firing_rate_func)

    # TODO improve the following way of finding the dimension
    dimension = input_dict['mu']['params']['K'].shape[0]
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
