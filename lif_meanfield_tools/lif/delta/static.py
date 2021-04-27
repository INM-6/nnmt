import numpy as np
from scipy.special import erfcx, dawsn, roots_legendre
from scipy.integrate import quad

from ..static import (_firing_rate_integration,
                      mean_input as _mean_input,
                      std_input as _std_input)

from ...utils import _check_and_store
from ... import ureg


prefix = 'lif.delta.'


@_check_and_store(prefix, ['firing_rates'])
def firing_rates(network):
    """
    Calculates stationary firing rates for delta shaped PSCs.

    Parameters:
    -----------
    network: lif_meanfield_tools.create.Network or child class instance.
        Network with the network parameters listed in the following.
    
    Network parameters:
    -------------------
    tau_m: float
        Membrane time constant in s.
    tau_r: float
        Refractory time in s.
    V_0_rel: float
        Relative reset potential in V.
    V_th_rel: float
        Relative threshold potential in V.
    K: np.array
        Indegree matrix.
    J: np.array
        Weight matrix in V.
    j: float
        Synaptic weight in V.
    nu_ext: float
        Firing rate of external input in Hz.
    K_ext: np.array
        Numbers of external input neurons to each population.
    g: float
        Relative inhibitory weight.
    nu_e_ext: float
        Firing rate of additional external excitatory Poisson input in Hz.
    nu_i_ext: float
        Firing rate of additional external inhibitory Poisson input in Hz.

    Returns:
    --------
    Quantity(np.array, 'hertz')
        Array of firing rates of each population in Hz.
    """
    list_of_firing_rate_params = ['tau_m', 'tau_r', 'V_th_rel', 'V_0_rel']
    list_of_input_params = ['K', 'J', 'tau_m', 'nu_ext', 'K_ext', 'J_ext',
                            'tau_m_ext']

    try:
        firing_rate_params = {key: network.network_params[key]
                              for key in list_of_firing_rate_params}
        input_params = {key: network.network_params[key]
                        for key in list_of_input_params}
    except KeyError as param:
        raise RuntimeError(f'You are missing {param} for this calculation.')
    
    return _firing_rate_integration(_firing_rate,
                                    firing_rate_params,
                                    input_params) * ureg.Hz


def _firing_rate(tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma):
    """
    Calculates stationary firing rate for delta shaped PSCs.
    
    For more documentation see `firing_rates`.
    
    Parameters:
    -----------
    tau_m: float
        Membrane time constant in s.
    tau_r: float
        Refractory time in s.
    V_0_rel: float or np.array
        Relative reset potential in V.
    V_th_rel: float or np.array
        Relative threshold potential in V.
    mu: float or np.array
        Mean input to population of neurons.
    sigma: float or np.array
        Standard deviation of input to population of neurons.
        
    Returns:
    float or np.array
        Firing rates in Hz.
    """
    y_th = (V_th_rel - mu) / sigma
    y_r = (V_0_rel - mu) / sigma
    y_th = np.atleast_1d(y_th)
    y_r = np.atleast_1d(y_r)
    tau_m = np.atleast_1d(tau_m)
    if len(tau_m) == 1:
        tau_m = np.ones(len(y_r)) * tau_m[0]
    tau_r = np.atleast_1d(tau_r)
    if len(tau_r) == 1:
        tau_r = np.ones(len(y_r)) * tau_r[0]
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
    params = {'tau_m': tau_m[mask_exc], 't_ref': tau_r[mask_exc],
              'gl_order': gl_order}
    nu[mask_exc] = _siegert_exc(y_th=y_th[mask_exc],
                                y_r=y_r[mask_exc], **params)
    params = {'tau_m': tau_m[mask_inh], 't_ref': tau_r[mask_inh],
              'gl_order': gl_order}
    nu[mask_inh] = _siegert_inh(y_th=y_th[mask_inh],
                                y_r=y_r[mask_inh], **params)
    params = {'tau_m': tau_m[mask_interm], 't_ref': tau_r[mask_interm],
              'gl_order': gl_order}
    nu[mask_interm] = _siegert_interm(y_th=y_th[mask_interm],
                                      y_r=y_r[mask_interm], **params)
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
    return (b - a) * np.sum(w * erfcx((b - a) * x / 2 + (b + a) / 2),
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
    Int = (2 * dawsn(y_th) - 2
           * np.exp(y_r**2 - y_th**2) * dawsn(y_r))
    Int -= e_V_th_2 * _erfcx_integral(y_r, y_th, gl_order)
    return e_V_th_2 / (e_V_th_2 * t_ref + tau_m * np.sqrt(np.pi) * Int)


def _siegert_interm(y_th, y_r, tau_m, t_ref, gl_order):
    """Calculate Siegert for y_r <= 0 <= y_th."""
    assert np.all((y_r <= 0) & (0 <= y_th))
    e_V_th_2 = np.exp(-y_th**2)
    Int = 2 * dawsn(y_th)
    Int += e_V_th_2 * _erfcx_integral(y_th, np.abs(y_r), gl_order)
    return e_V_th_2 / (e_V_th_2 * t_ref + tau_m * np.sqrt(np.pi) * Int)


@_check_and_store(prefix, ['mean_input'])
def mean_input(network):
    '''
    Calc mean inputs to populations as function of firing rates of populations.

    Following Fourcaud & Brunel (2002).
    
    Parameters:
    -----------
    network: lif_meanfield_tools.create.Network or child class instance.
        Network with the network parameters and previously calculated results
        listed in the following.
        
    Network results:
    ----------------
    nu: Quantity(np.array, 'hertz')
        Firing rates of populations in Hz.
    
    Network parameters:
    -------------------
    tau_m: float
        Membrane time constant in s.
    K: np.array
        Indegree matrix.
    J: np.array
        Weight matrix in V.
    j: float
        Synaptic weight in V.
    nu_ext: float
        Firing rate of external input in Hz.
    K_ext: np.array
        Numbers of external input neurons to each population.
    g: float
        Relative inhibitory weight.
    nu_e_ext: float
        Firing rate of additional external excitatory Poisson input in Hz.
    nu_i_ext: float
        Firing rate of additional external inhibitory Poisson input in Hz.

    Returns:
    --------
    Quantity(np.array, 'volt')
        Array of mean inputs to each population in V.
    '''
    return _mean_input(network, prefix)


@_check_and_store(prefix, ['std_input'])
def std_input(network):
    '''
    Calc standard deviation of inputs to populations.
    
    Following Fourcaud & Brunel (2002).
    
    Parameters:
    -----------
    network: lif_meanfield_tools.create.Network or child class instance.
        Network with the network parameters and previously calculated results
        listed in the following.
        
    Network results:
    ----------------
    nu: Quantity(np.array, 'hertz')
        Firing rates of populations in Hz.
    
    Network parameters:
    -------------------
    tau_m: float
        Membrane time constant in s.
    K: np.array
        Indegree matrix.
    J: np.array
        Weight matrix in V.
    j: float
        Synaptic weight in V.
    nu_ext: float
        Firing rate of external input in Hz.
    K_ext: np.array
        Numbers of external input neurons to each population.
    g: float
        Relative inhibitory weight.
    nu_e_ext: float
        Firing rate of additional external excitatory Poisson input in Hz.
    nu_i_ext: float
        Firing rate of additional external inhibitory Poisson input in Hz.

    Returns:
    --------
    Quantity(np.array, 'volt')
        Array of mean inputs to each population in V.
    '''
    return _std_input(network, prefix)
