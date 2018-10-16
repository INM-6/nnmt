from __future__ import print_function

import numpy as np
import pint
from scipy.special import zetac

from .__init__ import ureg
from . import aux_calcs


@ureg.wraps(ureg.Hz, (None, ureg.s, ureg.s,
                      ureg.s, ureg.mV, ureg.mV, None, ureg.mV,
                      ureg.mV, ureg.Hz, None))
def firing_rates(dimension, tau_m, tau_s, tau_r, V_0_rel, V_th_rel, K, J, j,
                 nu_ext, K_ext):
    '''
    Returns vector of population firing rates in Hz.

    Parameters:
    -----------
    dimension: Quantity(int, 'dimensionless')
        Number of populations.
    tau_m: Quantity(float, 'second')
        Membrane time constant.
    tau_s: Quantity(float, 'second')
        Synaptic time constant.
    tau_r: Quantity(float, 'second')
        Refractory time.
    V_0_rel: Quantity(float, 'millivolt')
        Relative reset potential.
    V_th_rel: Quantity(float, 'millivolt')
        Relative threshold potential.
    K: Quantity(np.ndarray, 'dimensionless')
        Indegree matrix.
    J: Quantity(np.ndarray, 'millivolt')
        Effective connectivity matrix.
    j: Quantity(float, 'millivolt')
        Effective connectivity weight.
    nu_ext: Quantity(float, 'hertz')
        Firing rate of external input.
    K_ext: Quantity(np.ndarray, 'dimensionless')
        Numbers of external input neurons to each population.

    Returns:
    --------
    Quantity(np.ndarray, 'hertz')
        Array of firing rates of each population in hertz.
    '''

    def rate_function(mu, sigma):
        """ calculates stationary firing rate with given parameters """
        return aux_calcs.nu0_fb433(tau_m, tau_s, tau_r, V_th_rel, V_0_rel, mu,
                                   sigma)

    def get_rate_difference(nu):
        """ calculate difference between new iteration step and previous one """
        ### new mean
        # contribution from within the network
        m0 = np.dot(K * J, nu) * tau_m
        # contribution from external sources
        m_ext = j * K_ext * nu_ext * tau_m
        # add them up
        mu = m0 + m_ext

        ### new std
        # contribution from within the network to variance
        var0 = np.dot(K * J**2, nu) * tau_m
        # contribution from external sources to variance
        var_ext = j**2 * K_ext * nu_ext * tau_m
        # add them up
        var = var0 + var_ext
        # standard deviation is square root of variance
        sigma = np.sqrt(var)

        new_nu = np.array([x for x in list(map(rate_function, mu, sigma))])

        return -nu + new_nu

    # do iteration procedure, until stationary firing rates are found
    dt = 0.05
    y = np.zeros((2, int(dimension)))
    eps = 1.0
    while eps >= 1e-5:
        delta_y = get_rate_difference(y[0])
        y[1] = y[0] + delta_y*dt
        epsilon = (y[1] - y[0])
        eps = max(np.abs(epsilon))
        y[0] = y[1]

    return y[1]

@ureg.wraps(ureg.mV, (ureg.Hz, None, ureg.mV, ureg.mV, ureg.s,
                      ureg.Hz, None))
def mean(nu, K, J, j, tau_m, nu_ext, K_ext):
    '''
    Calc mean inputs to populations as function of firing rates of populations

    Following Fourcaud & Brunel (2002)

    Parameters:
    -----------
    nu: Quantity(np.ndarray, 'hertz')
        firing rates of populations
    K: Quantity(np.ndarray, 'dimensionless')
        indegree matrix
    J: Quantity(np.ndarray, 'millivolt')
        effective connectivity matrix
    j: Quantity(float, 'millivolt')
        effective connectivity weight
    tau_m: Quantity(float, 'millisecond')
        membrane time constant
    nu_ext: Quantity(float, 'hertz')
        firing rate of external input
    K_ext: Quantity(np.ndarray, 'dimensionless')
        numbers of external input neurons to each population

    Returns:
    --------
    Quantity(np.ndarray, 'millivolt')
        array of mean inputs to each population in millivolt
    '''

    # contribution from within the network
    m0 = np.dot(K * J, nu) * tau_m
    # contribution from external sources
    m_ext = j * K_ext * nu_ext * tau_m
    # add them up
    m = m0 + m_ext

    return m


@ureg.wraps(ureg.mV, (ureg.Hz, None, ureg.mV, ureg.mV, ureg.s,
                      ureg.Hz, None))
def standard_deviation(nu, K, J, j, tau_m, nu_ext, K_ext):
    '''
    Calc standard devs of inputs to populations as function of firing rates

    Following Fourcaud & Brunel (2002)

    Parameters:
    -----------
    nu: Quantity(np.ndarray, 'hertz')
        firing rates of populations
    K: Quantity(np.ndarray, 'dimensionless')
        indegree matrix
    J: Quantity(np.ndarray, 'millivolt')
        effective connectivity matrix
    j: Quantity(float, 'millivolt')
        effective connectivity weight
    tau_m: Quantity(float, 'millisecond')
        membrane time constant
    nu_ext: Quantity(float, 'hertz')
        firing rate of external input
    K_ext: Quantity(np.ndarray, 'dimensionless')
        numbers of external input neurons to each population

    Returns:
    --------
    Quantity(np.ndarray, 'millivolt')
        array of standard dev of inputs to each population in millivolt
    '''
    # contribution from within the network to variance
    var0 = np.dot(K * J**2, nu) * tau_m
    # contribution from external sources to variance
    var_ext = j**2 * K_ext * nu_ext * tau_m
    # add them up
    var = var0 + var_ext
    # standard deviation is square root of variance
    std = np.sqrt(var)

    return std


@ureg.wraps(ureg.Hz/ureg.mV, (ureg.mV, ureg.mV, ureg.s, ureg.s, ureg.s,
                              ureg.mV, ureg.mV, ureg.Hz))
def transfer_function_1p_taylor(mu, sigma, tau_m, tau_s, tau_r, V_th_rel,
                                V_0_rel, omega):
    """
    Calcs value of transfer func for one population at given frequency omega.

    The calculation is done according to Eq. 93 in Schuecker et al (2014).

    Parameters:
    -----------
    mu: Quantity(float, 'millivolt')
        Mean neuron activity of one population in mV.
    sigma: Quantity(float, 'millivolt')
        Standard deviation of neuron activity of one population in mV.
    tau_m: Quantity(float, 'millisecond')
        Membrane time constant.
    tau_s: Quantity(float, 'millisecond')
        Synaptic time constant.
    tau_r: Quantity(float, 'millisecond')
        Refractory time.
    V_th_rel: Quantity(float, 'millivolt')
        Relative threshold potential.
    V_0_rel: Quantity(float, 'millivolt')
        Relative reset potential.
    omega: Quantity(flaot, 'hertz')
        Input frequency to population.

    Returns:
    --------
    Quantity(float, 'hertz/millivolt')
    """

    # for frequency zero the exact expression is given by the derivative of
    # f-I-curve
    if np.abs(omega- 0.) < 1e-15:
        return aux_calcs.d_nu_d_mu_fb433(tau_m, tau_s, tau_r, V_th_rel, V_0_rel,
                                         mu, sigma)
    else:
        nu0 = aux_calcs.nu_0(tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma)
        nu0_fb = aux_calcs.nu0_fb433(tau_m, tau_s, tau_r, V_th_rel, V_0_rel, mu,
                                     sigma)
        x_t = np.sqrt(2.) * (V_th_rel - mu) / sigma
        x_r = np.sqrt(2.) * (V_0_rel - mu) / sigma
        z = complex(-0.5, complex(omega * tau_m))
        alpha = np.sqrt(2) * abs(zetac(0.5) + 1)
        k = np.sqrt(tau_s / tau_m)
        A = alpha * tau_m * nu0 * k / np.sqrt(2)
        a0 = aux_calcs.Psi_x_r(z, x_t, x_r)
        a1 = aux_calcs.dPsi_x_r(z, x_t, x_r) / a0
        a3 = A / tau_m / nu0_fb * (-a1**2 + aux_calcs.d2Psi_x_r(z, x_t, x_r)/a0)
        result = (np.sqrt(2.) / sigma * nu0_fb / complex(1., omega * tau_m)* (a1 + a3))
        return result


@ureg.wraps(ureg.Hz/ureg.mV, (ureg.mV, ureg.mV, ureg.s, ureg.s, ureg.s,
                              ureg.mV, ureg.mV, ureg.Hz))
def transfer_function_1p_shift(mu, sigma, tau_m, tau_s, tau_r, V_th_rel,
                               V_0_rel, omega):
    """
    Calcs value of transfer func for one population at given frequency omega.

    Calculates transfer function according to $\tilde{n}$ in Schuecker et al.
    (2015). The expression is to first order equivalent to
    `transfer_function_1p_taylor`. Since the underlying theory is correct to
    first order, the two expressions are exchangeable.

    Parameters:
    -----------
    mu: Quantity(float, 'millivolt')
        Mean neuron activity of one population in mV.
    sigma: Quantity(float, 'millivolt')
        Standard deviation of neuron activity of one population in mV.
    tau_m: Quantity(float, 'millisecond')
        Membrane time constant.
    tau_s: Quantity(float, 'millisecond')
        Synaptic time constant.
    tau_r: Quantity(float, 'millisecond')
        Refractory time.
    V_th_rel: Quantity(float, 'millivolt')
        Relative threshold potential.
    V_0_rel: Quantity(float, 'millivolt')
        Relative reset potential.
    omega: Quantity(flaot, 'hertz')
        Input frequency to population.

    Returns:
    --------
    Quantity(float, 'hertz/millivolt')
    """

    # effective threshold and reset
    alpha = np.sqrt(2) * abs(zetac(0.5) + 1)
    V_th_rel += sigma * alpha / 2. * np.sqrt(tau_s / tau_m)
    V_0_rel += sigma * alpha / 2. * np.sqrt(tau_s / tau_m)

    # for frequency zero the exact expression is given by the derivative of
    # f-I-curve
    if np.abs(omega - 0.) < 1e-15:
        return aux_calcs.d_nu_d_mu(tau_m, tau_s, tau_r, V_th_rel, V_0_rel, mu,
                                   sigma)
    else:
        nu = aux_calcs.nu_0(tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma)

        x_t = np.sqrt(2.) * (V_th_rel - mu) / sigma
        x_r = np.sqrt(2.) * (V_0_rel - mu) / sigma
        z = complex(-0.5, complex(omega * tau_m))

        frac = aux_calcs.dPsi_x_r(z, x_t, x_r) / aux_calcs.Psi_x_r(z, x_t, x_r)

        return (np.sqrt(2.) / sigma * nu
                / (1. + complex(0., complex(omega*tau_m))) * frac)


def transfer_function(mu, sigma, tau_m, tau_s, tau_r, V_th_rel, V_0_rel,
                      dimension, omegas):
    """
    Returns transfer functions for all populations.

    Parameters:
    -----------
    mu: Quantity(float, 'millivolt')
        Mean neuron activity of one population in mV.
    sigma: Quantity(float, 'millivolt')
        Standard deviation of neuron activity of one population in mV.
    tau_m: Quantity(float, 'millisecond')
        Membrane time constant.
    tau_s: Quantity(float, 'millisecond')
        Synaptic time constant.
    tau_r: Quantity(float, 'millisecond')
        Refractory time.
    V_th_rel: Quantity(float, 'millivolt')
        Relative threshold potential.
    V_0_rel: Quantity(float, 'millivolt')
        Relative reset potential.
    omegas: Quantity(np.ndarray, 'hertz')
        Input frequencies to population.

    Returns:
    --------
    list of Quantities(np.nd.array, 'hertz/millivolt'):
        Returns one array for each population collected in a list. The arrays
        contain the values of the transfer function corresponding to the
        given omegas.
    """

    transfer_functions = [[transfer_function_1p_shift(mu[i], sigma[i], tau_m,
                                                      tau_s, tau_r, V_th_rel,
                                                      V_0_rel, omega)
                           for omega in omegas]
                          for i in range(dimension)]

    # convert list of list of quantities to list of quantities containing np.ndarray
    tf_magnitudes = [np.array([tf.magnitude for tf in tf_population])
                     for tf_population in transfer_functions]
    tf_unit = transfer_functions[0][0].units

    return tf_magnitudes * tf_unit


def delay_dist_matrix(dimension, Delay, Delay_sd, delay_dist, omega):
    '''
    Calcs matrix of delay distribution specific pre-factors at frequency omega.

    ???
    Assumes lower boundary for truncated Gaussian distributed delays to be zero
    (exact would be dt, the minimal time step).

    We had to define the subfunctions ddm_none, ddm_tg and ddm_g, because one
    cannot pass a string to a function decorated with ureg.wraps. So, that is
    how we bypass this issue. It is not very elegant though.

    Parameters:
    -----------
    dimension: Quantity(int, 'dimensionless')
        Dimension of the system / number of populations'
    Delay: Quantity(np.ndarray, 's')
        Delay matrix.
    Delay_sd: Quantity(np.ndarray, 's')
        Delay standard deviation matrix.
    delay_dist: str
        String specifying delay distribution.
    omega: float
        Frequency.

    Returns:
    --------
    Quantity(nd.array, 'dimensionless')
        Matrix of delay distribution specific pre-factors at frequency omega.
    '''

    if delay_dist == 'none':
        @ureg.wraps(ureg.dimensionless, (ureg.Hz, ureg.s, None))
        def ddm_none(omega, Delay, dimension):
            D = np.ones((int(dimension),int(dimension)))
            return D*np.exp(-complex(0,omega)*Delay)
        return ddm_none(omega, Delay, dimension)

    elif delay_dist == 'truncated_gaussian':
        @ureg.wraps(ureg.dimensionless, (ureg.s, ureg.s, ureg.Hz))
        def ddm_tg(Delay, Delay_sd, omega):
            a0 = aux_calcs.Phi(-Delay/Delay_sd+1j*omega*Delay_sd)
            a1 = aux_calcs.Phi(-Delay/Delay_sd)
            b0 = np.exp(-0.5*np.power(Delay_sd*omega,2))
            b1 = np.exp(-complex(0,omega)*Delay)
            return (1.0-a0)/(1.0-a1)*b0*b1
        return ddm_tg(Delay, Delay_sd, omega)

    elif delay_dist == 'gaussian':
        @ureg.wraps(ureg.dimensionless, (ureg.s, ureg.s, ureg.Hz))
        def ddm_g(Delay, Delay_sd, omega):
            b0 = np.exp(-0.5*np.power(Delay_sd*omega,2))
            b1 = np.exp(-complex(0,omega)*Delay)
            return b0*b1
        return ddm_g(Delay, Delay_sd, omega)


@ureg.wraps(ureg.dimensionless, (ureg.Hz/ureg.mV, ureg.dimensionless, ureg.mV,
                                 ureg.s, ureg.s, None, ureg.Hz))
def sensitivity_measure(transfer_function, delay_distr_matrix, J, tau_m, tau_s,
                        dimension, omega):
    """
    Calculates sensitivity measure as in Eq. 21 in Bos et al. (2015).

    Parameters:
    -----------
    # freq: frequency in Hz
    # Keyword arguments:
    # index: specifies index of eigenmode, default: None
    #        if set to None the dominant eigenmode is assumed

    Returns:
    --------

    """

    H = tau_m * transfer_function / complex(1, omega*tau_s)
    H = np.hstack([H for i in range(int(dimension))])
    H = np.transpose(H.reshape(int(dimension),int(dimension)))
    MH = H*J*delay_distr_matrix

    e, U = np.linalg.eig(MH)
    U_inv = np.linalg.inv(U)
    index = None
    if index is None:
        # find eigenvalue closest to one
        index = np.argmin(np.abs(e-1))
    T = np.outer(U_inv[index],U[:,index])
    T /= np.dot(U_inv[index],U[:,index])
    T *= MH

    return T
