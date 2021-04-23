import numpy as np

from scipy.special import erf, zetac, lambertw, erfcx, dawsn, roots_legendre
import numpy as np
import mpmath
from scipy.integrate import quad

from ... import ureg
from ...utils import (check_if_positive,
                      check_for_valid_k_in_fast_synaptic_regime)

from ..static import _firing_rate_integration


def firing_rates(network, method='scef'):
    list_of_firing_rate_params = ['tau_m', 'tau_r', 'V_th_rel', 'V_0_rel']
    list_of_input_params = ['K', 'J', 'j', 'tau_m', 'nu_ext', 'K_ext', 'g',
                            'nu_e_ext', 'nu_i_ext']
    try:
        firing_rate_params = {key: network.network_params[key]
                              for key in list_of_firing_rate_params}
        input_params = {key: network.network_params[key]
                        for key in list_of_input_params}
    except KeyError as param:
        print(f"You are missing {param} for calculating the firing rate!\n"
              "Have a look into the documentation for more details on 'lif'"
              " parameters.")
        
    firing_rate_params['method'] = method
        
    return _firing_rate_integration(_firing_rate,
                                    firing_rate_params,
                                    input_params)


def _firing_rate(tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma, method):
    if method == 'scef':
        return _firing_rate_scef(tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma)
    elif method == 'hds2017':
        # we should vectorize _firing_rate_hds2017 as some point
        mu = np.atleast_1d(mu)
        sigma = np.atleast_1d(sigma)
        V_th_rel = np.atleast_1d(V_th_rel)
        V_0_rel = np.atleast_1d(V_0_rel)
        nu = np.zeros(len(mu)) * ureg.Hz
        for i, (mu, sigma, V_th_rel, V_0_rel) in enumerate(zip(mu, sigma, V_th_rel, V_0_rel)):
            nu[i] =_firing_rate_hds2017(tau_m, tau_r, V_th_rel, V_0_rel,
                                        mu, sigma)
        return nu


def _firing_rate_scef(tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma):
    """
    Calculates stationary firing rates for delta shaped PSCs.

    Parameters:
    -----------
    tau_m: float
        Membrane time constant in seconds.
    tau_r: float
        Refractory time in seconds.
    V_th_rel: float
        Relative threshold potential in mV.
    V_0_rel: float
        Relative reset potential in mV.
    mu: float
        Mean neuron activity in mV.
    sigma: float
        Standard deviation of neuron activity in mV.
    method: str
        The method used for numerical integration of the Siegert formula.
        Options:
        - 'scef' (using the Scaled Complementary Error Function)
        - 'hds2017' (see appendix A.1. in Hahne, Dahmen, Schuecker et al. 2017)

    Returns:
    --------
    float:
        Stationary firing rate in Hz.
    """
    y_th = (V_th_rel - mu) / sigma
    y_r = (V_0_rel - mu) / sigma
    y_th = np.atleast_1d(y_th)
    y_r = np.atleast_1d(y_r)
    assert y_th.shape == y_r.shape
    assert y_th.ndim == y_r.ndim == 1
    if np.any(V_th_rel - V_0_rel < 0):
        raise ValueError('V_th should be larger than V_0!')

    # determine order of quadrature
    params = {'start_order': 10, 'epsrel': 1e-12, 'maxiter': 10}
    gl_order = _get_erfcx_integral_gl_order(y_th=y_th, y_r=y_r, **params)

    # separate domains
    mask_exc = y_th < 0
    mask_inh = 0 < y_r
    mask_interm = (y_r <= 0) & (0 <= y_th)

    # calculate siegert
    nu = np.zeros(shape=y_th.shape)
    params = {'tau_m': tau_m, 't_ref': tau_r, 'gl_order': gl_order}
    nu[mask_exc] = temp = _siegert_exc(y_th=y_th[mask_exc],
                                       y_r=y_r[mask_exc], **params)
    nu[mask_inh] = _siegert_inh(y_th=y_th[mask_inh],
                                y_r=y_r[mask_inh], **params)
    nu[mask_interm] = _siegert_interm(y_th=y_th[mask_interm],
                                      y_r=y_r[mask_interm], **params)
    # unit is stripped when quantity returned from _siegert_... is assigned to
    # elements in nu, so here we add it again
    try:
        nu = nu * temp.units
    except AttributeError:
        pass
    
    # convert back to scalar if only one value calculated
    if nu.shape == (1,):
        return nu.item(0)
    else:
        return nu
    
    
def _get_erfcx_integral_gl_order(y_th, y_r, start_order, epsrel, maxiter):
    """Determine order of Gauss-Legendre quadrature for erfcx integral."""
    # determine maximal integration range
    a = min(np.abs(y_th).min(), np.abs(y_r).min())
    b = max(np.abs(y_th).max(), np.abs(y_r).max())

    # adaptive quadrature from scipy.integrate for comparison
    I_quad = quad(erfcx, a, b, epsabs=0, epsrel=epsrel)[0]

    # increase order to reach desired accuracy
    order = start_order
    for _ in range(maxiter):
        I_gl = _erfcx_integral(a, b, order=order)[0]
        rel_error = np.abs(I_gl / I_quad - 1)
        if rel_error < epsrel:
            return order
        else:
            order *= 2
    msg = f'Quadrature search failed to converge after {maxiter} iterations. '
    msg += f'Last relative error {rel_error:e}, desired {epsrel:e}.'
    raise RuntimeError(msg)


def _erfcx_integral(a, b, order):
    """Fixed order Gauss-Legendre quadrature of erfcx from a to b."""
    assert np.all(a >= 0) and np.all(b >= 0)
    x, w = roots_legendre(order)
    x = x[:, np.newaxis]
    w = w[:, np.newaxis]
    return (b - a) * np.sum(w * _pint_erfcx((b - a) * x / 2 + (b + a) / 2),
                            axis=0) / 2


def _siegert_exc(y_th, y_r, tau_m, t_ref, gl_order):
    """Calculate Siegert for y_th < 0."""
    assert np.all(y_th < 0)
    Int = _erfcx_integral(np.abs(y_th), np.abs(y_r), gl_order)
    return 1 / (t_ref + tau_m * np.sqrt(np.pi) * Int)


def _siegert_inh(y_th, y_r, tau_m, t_ref, gl_order):
    """Calculate Siegert for 0 < y_th."""
    assert np.all(0 < y_r)
    e_V_th_2 = np.exp(-y_th**2)
    Int = (2 * _pint_dawsn(y_th) - 2
           * np.exp(y_r**2 - y_th**2) * _pint_dawsn(y_r))
    Int -= e_V_th_2 * _erfcx_integral(y_r, y_th, gl_order)
    return e_V_th_2 / (e_V_th_2 * t_ref + tau_m * np.sqrt(np.pi) * Int)


def _siegert_interm(y_th, y_r, tau_m, t_ref, gl_order):
    """Calculate Siegert for y_r <= 0 <= y_th."""
    assert np.all((y_r <= 0) & (0 <= y_th))
    e_V_th_2 = np.exp(-y_th**2)
    Int = 2 * _pint_dawsn(y_th)
    Int += e_V_th_2 * _erfcx_integral(y_th, np.abs(y_r), gl_order)
    return e_V_th_2 / (e_V_th_2 * t_ref + tau_m * np.sqrt(np.pi) * Int)


def _pint_erfcx(x):
    try:
        return erfcx(x)
    except TypeError:
        return erfcx(x.magnitude)


def _pint_dawsn(x):
    try:
        return dawsn(x)
    except TypeError:
        return dawsn(x.magnitude) * x.units


def _firing_rate_hds2017(tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma):
    if mu <= V_th_rel - 0.05 * abs(V_th_rel):
        return _siegert1(tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma)
    else:
        return _siegert2(tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma)


def _siegert1(tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma):
    """
    Calculates stationary firing rates for delta shaped PSCs for mu < V_th_rel.

    Parameters:
    -----------
    tau_m: float
        Membrane time constant in seconds.
    tau_r: float
        Refractory time in seconds.
    V_th_rel: float
        Relative threshold potential in mV.
    V_0_rel: float
        Relative reset potential in mV.
    mu: float
        Mean neuron activity in mV.
    sigma:
        Standard deviation of neuron activity in mV.

    Returns:
    --------
    float:
        Stationary firing rate in Hz.
    """
    pos_parameters = [tau_m, tau_r, sigma]
    pos_parameter_names = ['tau_m', 'tau_r', 'sigma']
    check_if_positive(pos_parameters, pos_parameter_names)

    if V_th_rel < V_0_rel:
        raise ValueError('V_th should be larger than V_0!')
    if mu > V_th_rel - 0.05 * abs(V_th_rel):
        raise ValueError('mu should be smaller than V_th-V_0! Use _siegert2 if'
                         ' mu > (V_th-V_0).')

    y_th = (V_th_rel - mu) / sigma
    y_r = (V_0_rel - mu) / sigma

    def integrand(u):
        if u == 0:
            return np.exp(-y_th**2) * 2 * (y_th - y_r)
        else:
            return np.exp(-(u - y_th)**2) * (1.0 - np.exp(2 * (y_r - y_th)
                                                          * u)) / u
    # find lower bound of integration, such that integrand is smaller than
    # 1e-12 at lower bound
    lower_bound = y_th
    err_dn = 1.0
    while err_dn > 1e-12 and lower_bound > 1e-16:
        err_dn = integrand(lower_bound)
        if err_dn > 1e-12:
            lower_bound /= 2

    # find upper bound of integration, such that integrand is smaller than
    # 1e-12 at lower bound
    upper_bound = y_th
    err_up = 1.0
    while err_up > 1e-12:
        err_up = integrand(upper_bound)
        if err_up > 1e-12:
            upper_bound *= 2

    # check preventing overflow
    if y_th >= 20:
        out = 0.
    if y_th < 20:
        out = 1.0 / (tau_r + np.exp(y_th**2)
                     * quad(integrand, lower_bound, upper_bound)[0] * tau_m)

    return out


def _siegert2(tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma):
    """
    Calculates stationary firing rates for delta shaped PSCs for mu > V_th_rel.

    Parameters:
    -----------
    tau_m: float
        Membrane time constant in seconds.
    tau_r: float
        Refractory time in seconds.
    V_th_rel: float
        Relative threshold potential in mV.
    V_0_rel: float
        Relative reset potential in mV.
    mu: float
        Mean neuron activity in mV.
    sigma:
        Standard deviation of neuron activity in mV.

    Returns:
    --------
    float:
        Stationary firing rate in Hz.
    """
    pos_parameters = [tau_m, tau_r, sigma]
    pos_parameter_names = ['tau_m', 'tau_r', 'sigma']
    check_if_positive(pos_parameters, pos_parameter_names)

    if V_th_rel < V_0_rel:
        raise ValueError('V_th should be larger than V_0!')
    # why this threshold?
    if mu < V_th_rel - 0.05 * abs(V_th_rel):
        raise ValueError('mu should be bigger than V_th-V_0 - 0.05 * '
                         'abs(V_th_rel)! Use siegert1 if mu < (V_th-V_0).')

    y_th = (V_th_rel - mu) / sigma
    y_r = (V_0_rel - mu) / sigma

    def integrand(u):
        if u == 0:
            return 2 * (y_th - y_r)
        else:
            return (np.exp(2 * y_th * u - u**2) - np.exp(2 * y_r * u - u**2)
                    ) / u

    upper_bound = 1.0
    err = 1.0
    while err > 1e-12:
        err = integrand(upper_bound)
        upper_bound *= 2

    return 1.0 / (tau_r + quad(integrand, 0.0, upper_bound)[0] * tau_m)
