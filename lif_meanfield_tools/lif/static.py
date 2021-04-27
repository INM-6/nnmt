from functools import partial
import numpy as np
import scipy.integrate as sint

from .. import ureg


def _firing_rate_integration(firing_rate_func, firing_rate_params,
                             input_params):
    """
    Solves the self-consistent eqs for firing rates, mean and std of input.
    
    Starts with a zero firing rate, calculates mean and std of the input and
    then calculates the firing rate again. This iterative procedure is repeated
    until the firing rates converge or an upper interation limit is reached.
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
    list_of_params = ['K', 'J', 'tau_m', 'nu_ext', 'K_ext', 'J_ext',
                      'tau_m_ext']
    try:
        params = {key: network.network_params[key] for key in list_of_params}
    except KeyError as param:
        raise RuntimeError(f'You are missing {param} for this calculation.')
    
    return input_func(rates, **params) * ureg.V
    

def _mean_input(nu, J, K, tau_m, nu_ext=0, J_ext=0, K_ext=0, tau_m_ext=0):
    """ Compute mean input without quantities. """
    # contribution from within the network
    m0 = np.dot(K * J, tau_m * nu)
    # contribution from external sources
    m_ext = np.dot(K_ext * J_ext, tau_m_ext * nu_ext)
    # add them up
    m = m0 + m_ext
    return m


def _std_input(nu, J, K, tau_m, nu_ext=0, J_ext=0, K_ext=0, tau_m_ext=0):
    """ Compute standard deviation of input without quantities. """
    # contribution from within the network to variance
    var0 = np.dot(K * J**2, tau_m * nu)
    # contribution from external sources to variance
    var_ext = np.dot(K_ext * J_ext**2, tau_m_ext * nu_ext)
    # add them up
    var = var0 + var_ext
    # standard deviation is square root of variance
    return np.sqrt(var)
