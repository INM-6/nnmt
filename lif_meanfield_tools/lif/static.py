import numpy as np
import scipy.integrate as sint

from .. import ureg


def _firing_rate_integration(firing_rate_func, firing_rate_params,
                             input_params):
    
    dimension = input_params['K'].shape[0]
    
    def get_rate_difference(_, nu):
        """
        Calculate difference between new iteration step and previous one.
        """
        try:
            nu.units
        except AttributeError:
            nu = nu * ureg.Hz
        mu = _mean(nu=nu, **input_params)
        sigma = _standard_deviation(nu=nu, **input_params)
        new_nu = firing_rate_func(mu=mu, sigma=sigma, **firing_rate_params)
        return -nu + new_nu

    # do iteration procedure, until stationary firing rates are found
    eps_tol = 1e-12
    t_max = 1000
    maxiter = 1000
    y0 = np.zeros(int(dimension))
    for _ in range(maxiter):
        sol = sint.solve_ivp(get_rate_difference, [0, t_max], y0,
                             t_eval=[t_max - 1, t_max], method='LSODA')
        assert sol.success is True
        eps = max(np.abs(sol.y[:, 1] - sol.y[:, 0]))
        if eps < eps_tol:
            return sol.y[:, 1]
        else:
            y0 = sol.y[:, 1]
    msg = f'Rate iteration failed to converge after {maxiter} iterations. '
    msg += f'Last maximum difference {eps:e}, desired {eps_tol:e}.'
    raise RuntimeError(msg)


def _mean(nu, K, J, j, tau_m, nu_ext, K_ext, g, nu_e_ext, nu_i_ext):
    """ Compute mean() without quantities. """
    try:
        nu.units
    except AttributeError:
        nu = nu * ureg.Hz
    # contribution from within the network
    m0 = tau_m * np.dot(K * J, nu)
    # contribution from external sources
    m_ext = tau_m * j * K_ext * nu_ext
    # contribution from additional excitatory and inhibitory Poisson input
    m_ext_add = tau_m * j * (nu_e_ext - g * nu_i_ext)
    # add them up
    m = m0 + m_ext + m_ext_add
    return m.to(ureg.mV)


def _standard_deviation(nu, K, J, j, tau_m, nu_ext, K_ext, g, nu_e_ext,
                        nu_i_ext):
    """ Compute standard_deviation() without quantities. """
    try:
        nu.units
    except AttributeError:
        nu = nu * ureg.Hz
    # contribution from within the network to variance
    var0 = tau_m * np.dot(K * J**2, nu)
    # contribution from external sources to variance
    var_ext = tau_m * j**2 * K_ext * nu_ext
    # contribution from additional excitatory and inhibitory Poisson input
    var_ext_add = tau_m * j**2 * (nu_e_ext + g**2 * nu_i_ext)
    # add them up
    var = var0 + var_ext + var_ext_add
    # standard deviation is square root of variance
    sigma = np.sqrt(var)
    return sigma.to(ureg.mV)
