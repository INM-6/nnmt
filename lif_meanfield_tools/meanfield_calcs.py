"""
In this module all the mean-field calculations are defined.

This module is called by network.py each time, a calculation is
executed.

Functions:
----------
firing_rates
mean
standard_deviation
transfer_function_1p_taylor
transfer_function_1p_shift
transfer_function
delay_dist_matrix
delay_dist_matrix_single
sensitivity_measure
power_spectra
eigen_spectra
additional_rates_for_fixed_input
fit_transfer_function
effective_coupling_strength
linear_interpolation_alpha
eigenvals_branches_rate
xi_of_k
solve_chareq_rate_boxcar
_standard_deviation
_mean
_effective_connectivity
_effective_connectivity_rate
_lambda_of_alpha_integral
_d_lambda_d_alpha
_xi_eff_s
_xi_eff_r
_d_xi_eff_s_d_lambda
_d_xi_eff_r_d_lambda
_solve_chareq_numerically_alpha
"""
from __future__ import print_function
import warnings
import numpy as np
import scipy.optimize as sopt
import scipy.integrate as sint
import scipy.misc as smisc
from scipy.special import zetac, erf


from . import ureg
from . import aux_calcs
from .utils import check_positive_params, check_k_in_fast_synaptic_regime


@check_positive_params
@ureg.wraps(ureg.Hz, (None, ureg.s, ureg.s, ureg.s, ureg.mV, ureg.mV, None,
                      ureg.mV, ureg.mV, ureg.Hz, None, None, ureg.Hz, ureg.Hz))
def firing_rates(dimension, tau_m, tau_s, tau_r, V_0_rel, V_th_rel, K, J, j,
                 nu_ext, K_ext, g, nu_e_ext, nu_i_ext):
    '''
    Returns vector of population firing rates in Hz.

    Parameters:
    -----------
    dimension: int
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
    K: np.ndarray
        Indegree matrix.
    J: Quantity(np.ndarray, 'millivolt')
        Weight matrix.
    j: Quantity(float, 'millivolt')
        Weight.
    nu_ext: Quantity(float, 'hertz')
        Firing rate of external input.
    K_ext: np.ndarray
        Numbers of external input neurons to each population.
    g: float
        relative inhibitory weight
    nu_e_ext: Quantity(float, 'hertz')
        firing rate of additional external excitatory Poisson input
    nu_i_ext: Quantity(float, 'hertz')
        firing rate of additional external inhibitory Poisson input

    Returns:
    --------
    Quantity(np.ndarray, 'hertz')
        Array of firing rates of each population in hertz.
    '''

    def rate_function(mu, sigma):
        """Calculate stationary firing rate with given parameters"""
        return aux_calcs._nu0_fb433(tau_m, tau_s, tau_r, V_th_rel, V_0_rel, mu,
                                    sigma)

    def get_rate_difference(nu):
        """Calculate difference between new iteration step and previous one"""
        # new mean
        mu = _mean(nu, K, J, j, tau_m, nu_ext, K_ext, g, nu_e_ext, nu_i_ext)

        # new std
        sigma = _standard_deviation(nu, K, J, j, tau_m, nu_ext, K_ext,
                                    g, nu_e_ext, nu_i_ext)

        new_nu = np.array([x for x in list(map(rate_function, mu, sigma))])

        return -nu + new_nu

    # do iteration procedure, until stationary firing rates are found
    dt = 0.05
    y = np.zeros((2, int(dimension)))
    eps = 1.0
    while eps >= 1e-5:
        delta_y = get_rate_difference(y[0])
        y[1] = y[0] + delta_y * dt
        epsilon = (y[1] - y[0])
        eps = max(np.abs(epsilon))
        y[0] = y[1]

    return y[1]


@check_positive_params
@ureg.wraps(ureg.mV, (ureg.Hz, None, ureg.mV, ureg.mV, ureg.s, ureg.Hz, None,
                      None, ureg.Hz, ureg.Hz))
def mean(nu, K, J, j, tau_m, nu_ext, K_ext, g, nu_e_ext, nu_i_ext):
    '''
    Calc mean inputs to populations as function of firing rates of populations

    Following Fourcaud & Brunel (2002)

    Parameters:
    -----------
    nu: Quantity(np.ndarray, 'hertz')
        firing rates of populations
    K: np.ndarray
        indegree matrix
    J: Quantity(np.ndarray, 'millivolt')
        Weight matrix
    j: Quantity(float, 'millivolt')
        Weight
    tau_m: Quantity(float, 'millisecond')
        membrane time constant
    nu_ext: Quantity(float, 'hertz')
        firing rate of external input
    K_ext: np.ndarray
        numbers of external input neurons to each population
    g: float
        relative inhibitory weight
    nu_e_ext: Quantity(float, 'hertz')
        firing rate of additional external excitatory Poisson input
    nu_i_ext: Quantity(float, 'hertz')
        firing rate of additional external inhibitory Poisson input

    Returns:
    --------
    Quantity(np.ndarray, 'millivolt')
        array of mean inputs to each population in millivolt
    '''
    return _mean(nu, K, J, j, tau_m, nu_ext, K_ext, g, nu_e_ext, nu_i_ext)


def _mean(nu, K, J, j, tau_m, nu_ext, K_ext, g, nu_e_ext, nu_i_ext):
    """ Compute mean() without quantities. """
    # contribution from within the network
    m0 = tau_m * np.dot(K * J, nu)
    # contribution from external sources
    m_ext = tau_m * j * K_ext * nu_ext
    # contribution from additional excitatory and inhibitory Poisson input
    m_ext_add = tau_m * j * (nu_e_ext - g * nu_i_ext)
    # add them up
    m = m0 + m_ext + m_ext_add

    return m


@check_positive_params
@ureg.wraps(ureg.mV, (ureg.Hz, None, ureg.mV, ureg.mV, ureg.s, ureg.Hz, None,
                      None, ureg.Hz, ureg.Hz))
def standard_deviation(nu, K, J, j, tau_m, nu_ext, K_ext, g, nu_e_ext,
                       nu_i_ext):
    '''
    Calc standard devs of inputs to populations as function of firing rates

    Following Fourcaud & Brunel (2002)

    Parameters:
    -----------
    nu: Quantity(np.ndarray, 'hertz')
        firing rates of populations
    K: np.ndarray
        indegree matrix
    J: Quantity(np.ndarray, 'millivolt')
        Weight  matrix
    j: Quantity(float, 'millivolt')
        Weight
    tau_m: Quantity(float, 'millisecond')
        membrane time constant
    nu_ext: Quantity(float, 'hertz')
        firing rate of external input
    K_ext: np.ndarray
        numbers of external input neurons to each population
    g: float
        relative inhibitory weight
    nu_e_ext: Quantity(float, 'hertz')
        firing rate of additional external excitatory Poisson input
    nu_i_ext: Quantity(float, 'hertz')
        firing rate of additional external inhibitory Poisson input

    Returns:
    --------
    Quantity(np.ndarray, 'millivolt')
        array of standard dev of inputs to each population in millivolt
    '''
    return _standard_deviation(nu, K, J, j, tau_m, nu_ext, K_ext,
                               g, nu_e_ext, nu_i_ext)


def _standard_deviation(nu, K, J, j, tau_m, nu_ext, K_ext, g, nu_e_ext,
                        nu_i_ext):
    """ Compute standard_deviation() without quantities. """
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
    return sigma


@check_positive_params
@check_k_in_fast_synaptic_regime
@ureg.wraps(ureg.Hz / ureg.mV, (ureg.mV, ureg.mV, ureg.s, ureg.s, ureg.s,
                                ureg.mV, ureg.mV, ureg.Hz, None))
def transfer_function_1p_taylor(mu, sigma, tau_m, tau_s, tau_r, V_th_rel,
                                V_0_rel, omega, synaptic_filter=True):
    """
    Calcs value of transfer func for one population at given frequency omega.

    The calculation is done according to Eq. 93 in Schuecker et al (2014).

    The difference here is that the linear response of the system is considered
    with respect to a perturbation of the input to the current I, leading to an
    additional low pass filtering 1/(1+i w tau_s).
    Compare with the second equation of Eq. 18 and the text below Eq. 29.

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
    if np.abs(omega - 0.) < 1e-15:
        result = aux_calcs.d_nu_d_mu_fb433(tau_m, tau_s, tau_r, V_th_rel,
                                           V_0_rel, mu, sigma)
    else:
        nu0 = aux_calcs.nu_0(tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma)
        nu0_fb = aux_calcs._nu0_fb433(tau_m, tau_s, tau_r, V_th_rel, V_0_rel,
                                      mu, sigma)
        x_t = np.sqrt(2.) * (V_th_rel - mu) / sigma
        x_r = np.sqrt(2.) * (V_0_rel - mu) / sigma
        z = complex(-0.5, complex(omega * tau_m))
        alpha = np.sqrt(2) * abs(zetac(0.5) + 1)
        k = np.sqrt(tau_s / tau_m)
        A = alpha * tau_m * nu0 * k / np.sqrt(2)
        a0 = aux_calcs.Psi_x_r(z, x_t, x_r)
        a1 = aux_calcs.dPsi_x_r(z, x_t, x_r) / a0
        a3 = A / tau_m / nu0_fb * (-a1**2
                                   + aux_calcs.d2Psi_x_r(z, x_t, x_r) / a0)
        result = (np.sqrt(2.) / sigma * nu0_fb
                  / complex(1., omega * tau_m) * (a1 + a3))

    if synaptic_filter:
        # additional low-pass filter due to perturbation to the input current
        return result / complex(1., omega * tau_s)
    return result


@check_positive_params
@check_k_in_fast_synaptic_regime
@ureg.wraps(ureg.Hz / ureg.mV, (ureg.mV, ureg.mV, ureg.s, ureg.s, ureg.s,
                                ureg.mV, ureg.mV, ureg.Hz, None))
def transfer_function_1p_shift(mu, sigma, tau_m, tau_s, tau_r, V_th_rel,
                               V_0_rel, omega, synaptic_filter=True):
    """
    Calcs value of transfer func for one population at given frequency omega.

    Calculates transfer function according to $\tilde{n}$ in Schuecker et al.
    (2015). The expression is to first order equivalent to
    `transfer_function_1p_taylor`. Since the underlying theory is correct to
    first order, the two expressions are exchangeable.

    The difference here is that the linear response of the system is considered
    with respect to a perturbation of the input to the current I, leading to an
    additional low pass filtering 1/(1+i w tau_s).
    Compare with the second equation of Eq. 18 and the text below Eq. 29.

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
    omega: Quantity(float, 'hertz')
        Input frequency to population.

    Returns:
    --------
    Quantity(float, 'hertz/millivolt')
    """
    return _transfer_function_1p_shift(mu, sigma, tau_m, tau_s, tau_r,
                                       V_th_rel, V_0_rel, omega,
                                       synaptic_filter)


def _transfer_function_1p_shift(mu, sigma, tau_m, tau_s, tau_r, V_th_rel,
                                V_0_rel, omega, synaptic_filter=True):
    """ Compute transfer_function_1p_shift() without quantities """
    # effective threshold and reset
    alpha = np.sqrt(2) * abs(zetac(0.5) + 1)
    V_th_rel += sigma * alpha / 2. * np.sqrt(tau_s / tau_m)
    V_0_rel += sigma * alpha / 2. * np.sqrt(tau_s / tau_m)

    # for frequency zero the exact expression is given by the derivative of
    # f-I-curve
    if np.abs(omega - 0.) < 1e-15:
        result = aux_calcs.d_nu_d_mu(tau_m, tau_r, V_th_rel, V_0_rel, mu,
                                     sigma)
    else:
        nu = aux_calcs.nu_0(tau_m, tau_r, V_th_rel, V_0_rel, mu, sigma)

        x_t = np.sqrt(2.) * (V_th_rel - mu) / sigma
        x_r = np.sqrt(2.) * (V_0_rel - mu) / sigma
        z = complex(-0.5, complex(omega * tau_m))

        frac = aux_calcs.dPsi_x_r(z, x_t, x_r) / aux_calcs.Psi_x_r(z, x_t, x_r)

        result = (np.sqrt(2.) / sigma * nu
                  / (1. + complex(0., complex(omega * tau_m))) * frac)

    if synaptic_filter:
        # additional low-pass filter due to perturbation to the input current
        return result / complex(1., omega * tau_s)
    return result


@check_positive_params
@check_k_in_fast_synaptic_regime
def transfer_function(mu, sigma, tau_m, tau_s, tau_r, V_th_rel, V_0_rel,
                      dimension, omegas, method='shift', synaptic_filter=True):
    """
    Returns transfer functions for all populations based on
    transfer_function_1p_shift() (default) or transfer_function_1p_taylor()

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
    dimension: int
        Number of populations.
    omegas: Quantity(np.ndarray, 'hertz')
        Input frequencies to population.
    method: str
        String specifying transfer function to use ('shift', 'taylor').

    Returns:
    --------
    list of Quantities(np.nd.array, 'hertz/millivolt'):
        Returns one array for each population collected in a list. The arrays
        contain the values of the transfer function corresponding to the
        given omegas.
    """

    if method == 'shift':
        transfer_functions = [
            [transfer_function_1p_shift(mu[i], sigma[i], tau_m, tau_s, tau_r,
                                        V_th_rel, V_0_rel, omega,
                                        synaptic_filter)
             for i in range(dimension)]
            for omega in omegas]
    if method == 'taylor':
        transfer_functions = [
            [transfer_function_1p_taylor(mu[i], sigma[i], tau_m, tau_s, tau_r,
                                         V_th_rel, V_0_rel, omega,
                                         synaptic_filter)
             for i in range(dimension)]
            for omega in omegas]

    # convert list of list of quantities to list of quantities containing
    # np.ndarray
    tf_magnitudes = np.array([np.array([tf.magnitude for tf in tf_population])
                              for tf_population in transfer_functions])
    tf_unit = transfer_functions[0][0].units

    return tf_magnitudes * tf_unit


@check_positive_params
@ureg.wraps(ureg.dimensionless, (None, ureg.s, ureg.s, None, ureg.Hz))
def delay_dist_matrix_single(dimension, Delay, Delay_sd, delay_dist, omega):
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
        Dimension of the system / number of populations
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
        D = np.ones((int(dimension), int(dimension)))
        return D * np.exp(-np.complex(0, omega) * Delay)

    elif delay_dist == 'truncated_gaussian':
        a0 = 0.5 * (1 + erf((-Delay / Delay_sd + 1j * omega * Delay_sd)
                            / np.sqrt(2)))
        a1 = 0.5 * (1 + erf((-Delay / Delay_sd) / np.sqrt(2)))
        b0 = np.exp(-0.5 * np.power(Delay_sd * omega, 2))
        b1 = np.exp(-np.complex(0, omega) * Delay)
        return (1.0 - a0) / (1.0 - a1) * b0 * b1

    elif delay_dist == 'gaussian':
        b0 = np.exp(-0.5 * np.power(Delay_sd * omega, 2))
        b1 = np.exp(-np.complex(0, omega) * Delay)
        return b0 * b1


def delay_dist_matrix(dimension, Delay, Delay_sd, delay_dist, omegas):
    """ Calculates delay distribution matrices for all omegas. """
    ddms = [delay_dist_matrix_single(dimension, Delay, Delay_sd, delay_dist,
                                     omega)
            for omega in omegas]

    # convert list of list of quantities to list of quantities containing
    # np.ndarray
    delay_dist_matrices = np.array([ddm.magnitude for ddm in ddms])
    ddm_unit = ddms[0].units

    return delay_dist_matrices * ddm_unit


def _effective_connectivity(omega, transfer_function, tau_m, J, K, dimension,
                            delay_term=1):
    """
    Effective connectivity.

    Parameters:
    -----------
    omega: float
        Input angular frequency to population in Hz.
    transfer_function: np.ndarray
        Transfer_function for given frequency omega in hertz/mV.
    tau_m: float
        Membrane time constant in s.
    J: np.ndarray
        Weight matrix in mV.
    K: np.ndarray
        Indegree matrix.
    dimension: int
        Number of populations.
    delay_term: 1 or np.ndarray
        optional delay_dist_matrix, unitless.

    Returns:
    --------
    np.ndarray
        Effective connectivity matrix.
    """
    # matrix of equal columns
    tf = np.tile(transfer_function, (dimension, 1)).T

    eff_conn = tau_m * J * K * tf * delay_term

    return eff_conn


def _effective_connectivity_rate(omega, tau, W_rate, delay_term=1):
    """
    Effective connectivity for rate model (first-order low-pass filter).

    Parameters:
    -----------
    omega: float
        Input angular frequency to population in Hz.
    tau: np.float
        Time constant in s.
    W: np.ndarray
        Dimensionless weight.
    delay_term: 1 or np.ndarray
        optional delay_dist_matrix, unitless.

    Returns:
    --------
    np.ndarray
        Effective connectivity matrix for rate model.
    """
    eff_conn = W_rate / (1. + 1j * omega * tau) * delay_term
    return eff_conn


@check_positive_params
@ureg.wraps(ureg.dimensionless, (ureg.Hz / ureg.mV, ureg.dimensionless,
                                 ureg.mV, None, ureg.s, ureg.s, None, ureg.Hz))
def sensitivity_measure(transfer_function, delay_dist_matrix, J, K, tau_m,
                        tau_s, dimension, omega):
    """
    Calculates sensitivity measure as in Eq. 21 in Bos et al. (2015).

    Parameters:
    -----------
    transfer_function: Quantity(np.ndarray, 'hertz/mV')
        Transfer_function for given frequency omega.
    delay_dist_matrix: Quantity(np.ndarray, 'dimensionless')
        Delay distribution matrix at given frequency.
    J: Quantity(np.ndarray, 'millivolt')
        Weight matrix.
    K: np.ndarray
        Indegree matrix.
    tau_m: Quantity(float, 'millisecond')
        Membrane time constant.
    tau_s: Quantity(float, 'millisecond')
        Synaptic time constant.
    dimension: int
        Number of populations.
    omega: Quantity(float, 'hertz')
        Input angular frequency to population.

    Returns:
    --------
    Quantity(np.ndarray, 'dimensionless')
        Sensitivity measure.
    """

    MH = _effective_connectivity(omega, transfer_function, tau_m, J, K,
                                 dimension, delay_dist_matrix)

    e, U = np.linalg.eig(MH)
    U_inv = np.linalg.inv(U)
    index = None
    if index is None:
        # find eigenvalue closest to one
        index = np.argmin(np.abs(e - 1))
    T = np.outer(U_inv[index], U[:, index])
    T /= np.dot(U_inv[index], U[:, index])
    T *= MH

    return T


@check_positive_params
@ureg.wraps(ureg.Hz, (ureg.s, ureg.s, None, ureg.mV, None, ureg.dimensionless,
                      None, ureg.Hz, ureg.Hz / ureg.mV, ureg.Hz))
def power_spectra(tau_m, tau_s, dimension, J, K, delay_dist_matrix, N,
                  firing_rates, transfer_function, omegas):
    """
    Calcs vector of power spectra for all populations at given frequencies.

    See: Eq. 18 in Bos et al. (2016)
    Shape of output: (len(populations), len(omegas))

    Parameters:
    -----------
    tau_m: Quantity(float, 'millisecond')
        Membrane time constant.
    tau_s: Quantity(float, 'millisecond')
        Synaptic time constant.
    dimension: int
        Number of populations.
    J: Quantity(np.ndarray, 'millivolt')
        Weight matrix.
    K: np.ndarray
        Indegree matrix.
    delay_dist_matrix: Quantity(np.ndarray, 'dimensionless')
        Delay distribution matrix at given frequency.
    N: np.ndarray
        Population sizes.
    firing_rates: Quantity(np.ndarray, 'hertz')
        Firing rates of the different populations.
    transfer_function: Quantity(np.ndarray, 'hertz/mV')
        Transfer_function for given frequency omega.
    omegas: Quantity(float, 'hertz')
        Input angular frequencies to population.

    Returns:
    --------
    Quantity(np.ndarray, 'hertz**2')
    """

    def power_spectra_single_freq(tau_m, tau_s, transfer_function, dimension,
                                  J, K, delay_dist_matrix, firing_rates, N,
                                  omega):
        """ Calculate power spectrum for single frequency. """

        MH = _effective_connectivity(omega, transfer_function, tau_m, J, K,
                                     dimension, delay_dist_matrix)

        Q = np.linalg.inv(np.identity(dimension) - MH)
        D = (np.diag(np.ones(dimension)) * firing_rates / N)
        C = np.dot(Q, np.dot(D, np.transpose(np.conjugate(Q))))
        spec = np.absolute(np.diag(C))
        return spec

    power = np.array([
        power_spectra_single_freq(tau_m, tau_s, transfer_function[i],
                                  dimension, J, K, delay_dist_matrix[i],
                                  firing_rates, N, omega)
        for i, omega in enumerate(omegas)])

    return np.transpose(power)


@check_positive_params
@ureg.wraps(ureg.dimensionless, (ureg.s, ureg.s, ureg.Hz / ureg.mV, None, None,
                                 ureg.mV, None, ureg.Hz, None, None))
def eigen_spectra(tau_m, tau_s, transfer_function, dimension,
                  delay_dist_matrix, J, K, omegas, quantity, matrix):
    """
    Calcs eigenvals, left and right eigenvecs of matrix at given frequency.

    Parameters:
    -----------
    tau_m: Quantity(float, 'millisecond')
        Membrane time constant.
    tau_s: Quantity(float, 'millisecond')
        Synaptic time constant.
    transfer_function: Quantity(np.ndarray, 'hertz/mV')
        Transfer_function for given frequency omega.
    dimension: int
        Number of populations.
    delay_dist_matrix: Quantity(np.ndarray, 'dimensionless')
        Delay distribution matrix at given frequency.
    J: Quantity(np.ndarray, 'millivolt')
        Weight matrix.
    K: np.ndarray
        Indegree matrix.
    omegas: Quantity(np.ndarray, 'hertz')
        Input angular frequency to population.
    quantity: str
        Specifies, what is returned. Options are 'eigvals', 'reigvecs',
        'leigvecs'.
    matrix: str
        String specifying which matrix is analysed. Options are the effective
        connectivity matrix 'MH', the propagator 'prop' and the inverse
        propagator 'prop_inv'.

    Returns:
    --------
    Quantity(np.ndarray, 'dimensionless')
        Either eigenvalues corresponding to given frequencies or right or left
        eigenvectors corresponding to given frequencies.
    """

    def eigen_spectra_single_freq(tau_m, tau_s, transfer_function, dimension,
                                  delay_dist_matrix, J, K, omega, matrix):

        MH = _effective_connectivity(omega, transfer_function, tau_m, J, K,
                                     dimension, delay_dist_matrix).magnitude

        if matrix == 'MH':
            eig, vr = np.linalg.eig(MH)
            vl = np.linalg.inv(vr)
            return eig, np.transpose(vr), vl

        Q = np.linalg.inv(np.identity(dimension) - MH)
        P = np.dot(Q, MH)
        if matrix == 'prop':
            eig, vr = np.linalg.eig(P)
        elif matrix == 'prop_inv':
            eig, vr = np.linalg.eig(np.linalg.inv(P))
        vl = np.linalg.inv(vr)

        return eig, np.transpose(vr), vl

    if quantity == 'eigvals':
        eig = [eigen_spectra_single_freq(tau_m, tau_s, transfer_function[i],
                                         dimension, delay_dist_matrix[i], J, K,
                                         omega, matrix)[0]
               for i, omega in enumerate(omegas)]
    elif quantity == 'reigvecs':
        eig = [eigen_spectra_single_freq(tau_m, tau_s, transfer_function[i],
                                         dimension, delay_dist_matrix[i], J, K,
                                         omega, matrix)[1]
               for i, omega in enumerate(omegas)]
    elif quantity == 'leigvecs':
        eig = [eigen_spectra_single_freq(tau_m, tau_s, transfer_function[i],
                                         dimension, delay_dist_matrix[i], J, K,
                                         omega, matrix)[2]
               for i, omega in enumerate(omegas)]

    return np.transpose(eig)


@check_positive_params
@ureg.wraps((ureg.Hz, ureg.Hz), (ureg.mV, ureg.mV, ureg.s, ureg.s, ureg.s,
                                 ureg.mV, ureg.mV,
                                 None, ureg.mV, ureg.mV, ureg.Hz, None, None))
def additional_rates_for_fixed_input(mu_set, sigma_set,
                                     tau_m, tau_s, tau_r,
                                     V_0_rel, V_th_rel,
                                     K, J, j, nu_ext, K_ext, g):
    """
    Calculate additional external excitatory and inhibitory Poisson input
    rates such that the input fixed by the mean and standard deviation
    is attained.
    Correction of equation E1 of:
    Helias M, Tetzlaff T, Diesmann M. Echoes in correlated neural systems.
    New J Phys. 2013;15(2):023002. doi:10.1088/1367-2630/15/2/023002.

    Parameters:
    -----------
    mean_input_set: Quantity(np.ndarray, 'mV')
        prescribed mean input for each population
    std_input_set: Quantity(np.ndarray, 'mV')
        prescribed standard deviation of input for each population
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
    K: np.ndarray
        Indegree matrix.
    J: Quantity(np.ndarray, 'millivolt')
        Weight matrix.
    j: Quantity(float, 'millivolt')
        Weight.
    nu_ext: Quantity(float, 'hertz')
        Firing rate of external input.
    K_ext: np.ndarray
        Numbers of external input neurons to each population.
    g: float

    Returns:
    --------
    nu_e_ext: Quantity(np.ndarray, 'hertz')
        additional external excitatory rate needed for fixed input
    nu_i_ext: Quantity(np.ndarray, 'hertz')
        additional external inhibitory rate needed for fixed input
    """
    target_rates = np.zeros(len(mu_set))
    for i in np.arange(len(mu_set)):
        # target rates for set mean and standard deviation of input
        target_rates[i] = aux_calcs._nu0_fb433(tau_m, tau_s, tau_r,
                                               V_th_rel, V_0_rel,
                                               mu_set[i], sigma_set[i])

    # additional external rates set to 0 for local-only contributions
    mu_loc = _mean(nu=target_rates, K=K, J=J, j=j, tau_m=tau_m,
                   nu_ext=nu_ext, K_ext=K_ext, g=g, nu_e_ext=0., nu_i_ext=0.)
    sigma_loc = _standard_deviation(nu=target_rates, K=K, J=J, j=j,
                                    tau_m=tau_m, nu_ext=nu_ext, K_ext=K_ext,
                                    g=g, nu_e_ext=0., nu_i_ext=0.)

    mu_temp = (mu_set - mu_loc) / (tau_m * j)
    sigma_temp_2 = (sigma_set**2 - sigma_loc**2) / (tau_m * j**2)

    nu_e_ext = (sigma_temp_2 + g * mu_temp) / (1. + g)
    nu_i_ext = (sigma_temp_2 - mu_temp) / (g * (1. + g))

    if np.any(np.array([nu_e_ext, nu_i_ext]) < 0):
        warn = ('Negative rate detected:\n\tnu_e_ext=' + str(nu_e_ext)
                + '\n\tnu_i_ext=' + str(nu_i_ext))
        warnings.warn(warn)

    return nu_e_ext, nu_i_ext


@check_positive_params
@ureg.wraps((ureg.ms, None, None, ureg.Hz / ureg.mV),
            (ureg.Hz / ureg.mV, ureg.Hz, ureg.s, ureg.mV, None))
def fit_transfer_function(transfer_function, omegas, tau_m, J, K):
    """
    Fit the absolute value of the LIF transfer function to the one of a
    first-order low-pass filter. Compute the time constants and weight matrices
    for an equivalent rate model.

    Parameters:
    -----------
    transfer_function: Quantity(np.ndarray, 'hertz/mV')
        Transfer_function for given frequencies omegas.
    omegas: Quantity(np.ndarray, 'hertz')
        Input frequencies to population.
    tau_m: Quantity(float, 'second')
        Membrane time constant.
    J: Quantity(np.ndarray, 'millivolt')
        Weight matrix.
    K: np.ndarray
        Indegree matrix.

    Returns:
    --------
    tau_rate: Quantity(np.ndarray, 's')
        time constants from fit
    W_rate: np.ndarray
        weight matrix from fit
    W_rate_sim: np.ndarray
        weight matrix from fit divided by indegrees
    tf_fit: np.ndarray
        fitted transfer function (columns = populations)
    """
    fit_tf, tau_rate, h0, err_tau, err_h0 = \
        _fit_transfer_function(transfer_function, omegas)

    W_rate_sim = h0 * tau_m * J
    W_rate = np.multiply(W_rate_sim, K)

    return tau_rate * 1.E3, W_rate, W_rate_sim, fit_tf


def _fit_transfer_function(transfer_function, omegas):
    """
    Fit transfer function.

    Parameters:
    -----------
    transfer_function: np.ndarray
        Transfer_function for given frequencies omegas.
    omegas: np.ndarray
        Frequencies in Hz.

    Reutrns:
    --------
    fit_tf: np.ndarray
        Fitted transfer function.
    tau_rate: np.ndarray
        Time constants from fit in s.
    h0: np.ndarray
        Offset of transfer function from fit in Hz/mV.
    err_tau: np.ndarray
        Relative fit error on time constant.
    err_h0: np.ndarray
        Relative fit error on offset.
    """
    def func(omega, tau, h0):
        return h0 / (1. + 1j * omega * tau)
    
    # absolute value for fitting
    def func_abs(omega, tau, h0):
        return np.abs(func(omega, tau, h0))

    fit_tf = np.zeros(np.shape(transfer_function), dtype=np.complex)
    dim = np.shape(transfer_function)[1]
    tau_rate = np.zeros(dim)
    h0 = np.zeros(dim)
    err_tau = np.zeros(dim)
    err_h0 = np.zeros(dim)

    bounds = [[0., -np.inf], [np.inf, np.inf]]
    for i in np.arange(np.shape(fit_tf)[1]):
        fitParams, fitCovariances = sopt.curve_fit(
            func_abs, omegas, np.abs(transfer_function[:, i]), bounds=bounds)
        tau_rate[i] = fitParams[0]
        h0[i] = fitParams[1]
        fit_tf[:, i] = func(omegas, tau_rate[i], h0[i])

        # 1 standard deviation
        fit_err = np.sqrt(np.diag(fitCovariances))
        # relative error
        err_tau[i] = fit_err[0] / tau_rate[i]
        err_h0[i] = fit_err[1] / h0[i]

    return fit_tf, tau_rate, h0, err_tau, err_h0


def scan_fit_transfer_function_mean_std_input(mean_inputs, std_inputs,
                                              tau_m, tau_s, tau_r,
                                              V_0_rel, V_th_rel, omegas):
    """
    Scan all combinations of mean_inputs and std_inputs: Compute and fit the
    transfer function for each case and return the relative fit errors on
    tau and h0.

    Parameters:
    -----------
    mean_inputs: Quantity(np.ndarray, 'mV')
        List of mean inputs to scan.
    std_inputs: Quantity(np.ndarray, 'mV')
        List of standard deviation of inputs to scan.
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
    omegas: Quantity(np.ndarray, 'hertz')
        Input angular frequencies to population.

    Returns:
    --------
    errs_tau: np.ndarray
        Rel. error on fitted tau for each combination of mean and std of input.
    errs_h0: np.ndarray
        Rel. error on fitted h0 for each combination of mean and std of input.
    """
    dims = (len(mean_inputs), len(std_inputs))
    errs_tau = np.zeros(dims)
    errs_h0 = np.zeros(dims)

    for i, mu in enumerate(mean_inputs):
        for j, sigma in enumerate(std_inputs):
            tfs = [[transfer_function_1p_shift(mu, sigma, tau_m,
                                               tau_s, tau_r, V_th_rel,
                                               V_0_rel, omega)]
                   for omega in omegas]
            tf_magnitudes = [[tf[0].magnitude] for tf in tfs]
            tf_unit = tfs[0][0].units
            transfer_function = tf_magnitudes * tf_unit

            fit_tf, tau_rate, h0, err_tau, err_h0 = (_fit_transfer_function(
                transfer_function.to(ureg.Hz / ureg.mV).magnitude,
                omegas.to(ureg.Hz).magnitude))

            errs_tau[i, j] = err_tau[0]
            errs_h0[i, j] = err_h0[0]
    return errs_tau, errs_h0


@check_positive_params
# @ureg.wraps(None, (ureg.s, ureg.s, ureg.s, ureg.mV, ureg.mV, ureg.mV, ureg.mV,
#                    ureg.mV))
def effective_coupling_strength(tau_m, tau_s, tau_r, V_0_rel, V_th_rel, J,
                                mean_input, std_input):
    """
    Compute effective coupling strength as the linear contribution of the
    derivative of nu_0 by input rate for low-pass-filtered synapses with tau_s.
    Effective threshold and reset from Fourcoud & Brunel 2002.

    Parameters:
    -----------
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
    K: np.ndarray
        Indegree matrix.
    J: Quantity(np.ndarray, 'millivolt')
        Weight matrix.
    j: Quantity(float, 'millivolt')
        Weight.
    nu_ext: Quantity(float, 'hertz')
        Firing rate of external input.
    K_ext: np.ndarray
        Numbers of external input neurons to each population.
    g: float
    """
    dim = len(mean_input)
    w_ecs = np.zeros((dim, dim))
    for pre in np.arange(dim):
        for post in np.arange(dim):
            # linear (mu) contribution
            w_ecs[post][pre] = aux_calcs.d_nu_d_nu_in_fb(
                tau_m, tau_s, tau_r, V_th_rel, V_0_rel, J[post][pre],
                mean_input[pre], std_input[pre])
    return w_ecs


@check_positive_params
@ureg.wraps((None, (1 / ureg.s).units, (1 / ureg.s).units, (1 / ureg.m).units,
             (1 / ureg.s).units, (1 / ureg.s).units),
            ((1 / ureg.m).units, None, ureg.s, None, ureg.m, ureg.s, ureg.s,
             ureg.mV, ureg.mV, ureg.s, ureg.s, ureg.s, ureg.mV, ureg.mV,
             ureg.mV, None, None))
def linear_interpolation_alpha(k_wavenumbers, branches, tau_rate, W_rate,
                               width, d_e, d_i, mean_inputs, std_inputs, tau_m,
                               tau_s, tau_r, V_0_rel, V_th_rel, J, K,
                               dimension):
    """
    Linear interpolation between analytically solved characteristic equation
    for linear rate model and equation solved for lif model.
    Eigenvalues lambda are computed by solving the characteristic equation
    numerically or by solving an integral.
    Requires a spatially organized network with boxcar connectivity profile.

    Parameters:
    -----------
    k_wavenumbers: Quantity(np.ndarray, '1/m')
        Range of wave numbers.
    branches: np.ndarray
        List of branches.
    tau_rate: Quantity(np.ndarray, 's')
        Time constants from fit.
    W_rate: np.ndarray
        Weights from fit.
    width: np.ndarray
        Spatial widths of boxcar connectivtiy profile.
    d_e: Quantity(float, 's')
        Excitatory delay.
    d_i: Quantity(float, 's')
        Inhibitory delay.
    mean_inputs: Quantity(np.ndarray, 'mV')
        List of mean inputs to scan.
    std_inputs: Quantity(np.ndarray, 'mV')
        List of standard deviation of inputs to scan.
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
    J: Quantity(np.ndarray, 'millivolt')
        Weight matrix.
    K: np.ndarray
        Indegree matrix.
    dimension: int
        Dimension of the system / number of populations.

    Returns:
    --------
    alphas: np.ndarray
    lambdas_chareq: Quantity(np.ndarray, '1/s')
    lambdas_integral: Quantity(np.ndarray, '1/s')
    k_eig_max: Quantity(float, '1/m')
    eigenval_max: Quantity(complex, '1/s')
    eigenvals: Quantity(np.ndarray, '1/s')
    """
    assert len(np.unique(tau_rate)) == 1, ('Linear interpolation requires '
                                           'equal tau_rate.')
    tau = tau_rate[0]
    assert d_e == d_i, 'Linear interpolation requires equal delay.'
    delay = d_e
    assert len(np.unique(mean_inputs)) == 1, ('Linear interpolation requires '
                                              'same mean input.')
    mu = mean_inputs[0]
    assert len(np.unique(mean_inputs)) == 1, ('Linear interpolation requires '
                                              'same std input.')
    sigma = std_inputs[0]

    # ground truth at alpha = 0 from rate model
    k_eig_max, idx_k_eig_max, eigenval_max, eigenvals = (
        eigenvals_branches_rate(k_wavenumbers, branches, tau, W_rate, width,
                                delay))

    # first alpha must be 0 for integrate.odeint! (initial condition)
    alphas = np.linspace(0, 1, 5)
    lambdas_integral = np.zeros((len(branches), (len(alphas))), dtype=complex)
    lambdas_chareq = np.zeros((len(branches), len(alphas)), dtype=complex)
    for i, branch in enumerate(branches):
        # evaluate all eigenvalues at k_eig_max (wavenumbers with largest real
        # part of eigenvalue from theory)
        lambda0 = eigenvals[i, idx_k_eig_max]
        print(branch, lambda0)
        # 1. solution by solving the characteristic equation numerically
        for j, alpha in enumerate(alphas):
            lambdas_chareq[i, j] = (
                _solve_chareq_numerically_alpha(lambda0, alpha, k_eig_max,
                                                delay, mu, sigma, tau_m, tau_s,
                                                tau_r, V_th_rel, V_0_rel, J, K,
                                                dimension, tau, W_rate, width))

        # 2. solution by solving the integral
        lambdas_integral[i, :] = _lambda_of_alpha_integral(
            alphas, lambda0, k_eig_max, delay, mu, sigma, tau_m, tau_s, tau_r,
            V_0_rel, V_th_rel, J, K, dimension, tau, W_rate, width)
    return (alphas, lambdas_chareq, lambdas_integral, k_eig_max, eigenval_max,
            eigenvals)


def eigenvals_branches_rate(k_wavenumbers, branches, tau, W_rate, width,
                            delay):
    """
    Compute in the linearized rate model for each branch the eigenvalues by
    solving the characteristic equation analytically.
    Requires a spatially organized network with boxcar connectivity profile.

    Parameters:
    -----------
    k_wavenumbers: np.ndarray
        Range of wave numbers in 1/m.
    branches: np.ndarray
        List of branches.
    tau: float
        Time constant from fit in s.
    W_rate: np.ndarray
        Weights from fit.
    width: np.ndarray
        Spatial widths of boxcar connectivtiy profile in m.
    delay: float
        Delay in s.

    Returns:
    --------
    k_eig_max: float
    idx_k_eig_max: float
    eigenval_max: complex
    eigenvals: np.ndarray
    """
    eigenvals = np.zeros((len(branches), len(k_wavenumbers)), dtype=complex)

    for i, branch in enumerate(branches):
        for j, k_wavenumber in enumerate(k_wavenumbers):
            eigenvals[i, j] = aux_calcs.solve_chareq_rate_boxcar(
                branch, k_wavenumber, tau, W_rate, width, delay)

    # index of eigenvalue with maximum real part
    idx_max = list(np.unravel_index(np.argmax(eigenvals.real),
                                    eigenvals.shape))

    # if max at branch -1, swap with 0
    if branches[idx_max[0]] == -1:
        # index of current branch -1
        idx_n1 = idx_max[0]
        # index of current branch 0
        idx_0 = list(branches).index(0)
        eigenvals[[idx_n1, idx_0], [idx_0, idx_n1]]
        idx_max[0] = idx_0

    eigenval_max = eigenvals[idx_max[0], idx_max[1]]
    k_eig_max = k_wavenumbers[idx_max[1]]
    idx_k_eig_max = idx_max[1]
    return k_eig_max, idx_k_eig_max, eigenval_max, eigenvals


def _lambda_of_alpha_integral(alphas, lambda0, k, delay, mu, sigma, tau_m,
                              tau_s, tau_r, V_0_rel, V_th_rel, J, K, dimension,
                              tau, W_rate, width):
    """
    Compute lambda of alpha by solving the integral.
    Requires a spatially organized network with boxcar connectivity profile.

    Parameters:
    -----------
    alphas: np.ndarray
        Range of interpolation parameters.
    lambda0: complex
        Guess for eigenvalue.
    k: float
        Wavenumber in 1/m.
    delay: float
        Delay in s.
    mu: float
        Mean input in mV.
    sigma: float
        Standard deviation of input in mV.
    tau_m: float
        Membrane time constant in s.
    tau_s: float
        Synaptic time constant in s.
    tau_r: float
        Refractory time in s.
    V_0_rel: float
        Relative reset potential in mV.
    V_th_rel: float
        Relative threshold potential in mV.
    J: np.ndarray
        Weight matrix in mV.
    K: np.ndarray
        Indegree matrix.
    dimension: int
        Dimension of the system / number of populations.
    tau_rate: float
        Time constant from fit in s.
    W_rate: np.ndarray
        Weights from fit.
    width: np.ndarray
        Spatial widths of boxcar connectivtiy profile in m.

    Returns:
    --------
    lambdas_of_alpha: list
    """
    assert alphas[0] == 0, 'First alpha must be 0!'
    lambda0_list = [lambda0.real, lambda0.imag]

    def derivative(lambda_list, a0):
        llist = complex(lambda_list[0], lambda_list[1])
        deriv = _d_lambda_d_alpha(llist, a0, k, delay, mu, sigma, tau_m, tau_s,
                                  tau_r, V_0_rel, V_th_rel, J, K, dimension,
                                  tau, W_rate, width)
        return [deriv.real, deriv.imag]

    llist = sint.odeint(func=derivative, y0=lambda0_list, t=alphas)

    lambdas_of_alpha = [complex(_l[0], _l[1]) for _l in llist]
    return lambdas_of_alpha


def _d_lambda_d_alpha(eval, alpha, k, delay, mu, sigma, tau_m, tau_s, tau_r,
                      V_0_rel, V_th_rel, J, K, dimension, tau, W_rate, width):
    """
    Compute the derivative of lambda with respect to alpha.
    Requires a spatially organized network with boxcar connectivity profile.

    Parameters:
    -----------
    eval: complex
        Eigenvalue.
    alpha: float
        Interpolation parameters.
    k: float
        Wavenumber in 1/m.
    delay: float
        Delay in s.
    mu: float
        Mean input in mV.
    sigma: float
        Standard deviation of input in mV.
    tau_m: float
        Membrane time constant in s.
    tau_s: float
        Synaptic time constant in s.
    tau_r: float
        Refractory time in s.
    V_0_rel: float
        Relative reset potential in mV.
    V_th_rel: float
        Relative threshold potential in mV.
    J: np.ndarray
        Weight matrix in mV.
    K: np.ndarray
        Indegree matrix.
    dimension: int
        Dimension of the system / number of populations.
    tau: float
        Time constant from fit in s.
    W_rate: np.ndarray
        Weights from fit.
    width: np.ndarray
        Spatial widths of boxcar connectivtiy profile in m.

    Returns:
    --------
    deriv: complex
    """
    xi_eff_s = _xi_eff_s(eval, k, mu, sigma, tau_m, tau_s, tau_r, V_th_rel,
                         V_0_rel, J, K, dimension, width)
    xi_eff_r = _xi_eff_r(eval, k, tau, W_rate, width)

    xi_eff_sr = xi_eff_s - xi_eff_r

    xi_eff_alpha = alpha * xi_eff_s + (1. - alpha) * xi_eff_r

    d_xi_eff_s_d_lambda = _d_xi_eff_s_d_lambda(eval, k, mu, sigma, tau_m,
                                               tau_s, tau_r, V_th_rel, V_0_rel,
                                               J, K, dimension, width)

    d_xi_eff_r_d_lambda = _d_xi_eff_r_d_lambda(eval, k, tau, W_rate, width)

    d_xi_eff_alpha_d_lambda = (alpha * d_xi_eff_s_d_lambda
                               + (1. - alpha) * d_xi_eff_r_d_lambda)

    nominator = xi_eff_sr
    denominator = d_xi_eff_alpha_d_lambda - delay * xi_eff_alpha

    deriv = - nominator / denominator
    return deriv


def _xi_eff_s(eval, k, mu, sigma, tau_m, tau_s, tau_r, V_th_rel, V_0_rel,
              J, K, dimension, width):
    """
    Compute xi_eff for the lif neuron model.
    Requires a spatially organized network with boxcar connectivity profile.

    Parameters:
    -----------
    eval: complex
        Eigenvalue.
    k: float
        Wavenumber in 1/m.
    mu: float
        Mean input in mV.
    sigma: float
        Standard deviation of input in mV.
    tau_m: float
        Membrane time constant in s.
    tau_s: float
        Synaptic time constant in s.
    tau_r: float
        Refractory time in s.
    V_th_rel: float
        Relative threshold potential in mV.
    V_0_rel: float
        Relative reset potential in mV.
    J: np.ndarray
        Weight matrix in mV.
    K: np.ndarray
        Indegree matrix.
    dimension: int
        Dimension of the system / number of populations.
    width: np.ndarray
        Spatial widths of boxcar connectivtiy profile in m.

    Returns:
    --------
    xi_eff_s: complex
    """
    omega = complex(0, -eval)
    transfer_func = _transfer_function_1p_shift(
        mu, sigma, tau_m, tau_s, tau_r, V_th_rel, V_0_rel, omega)

    MH_s = _effective_connectivity(omega, transfer_func, tau_m, J, K,
                                   dimension)
    P_hat = aux_calcs.p_hat_boxcar(k, width)
    xi_eff_s = aux_calcs.determinant(MH_s * P_hat)
    return xi_eff_s


def _xi_eff_r(eval, k, tau, W_rate, width):
    """
    Compute xi_eff for the linearized rate model.
    Requires a spatially organized network with boxcar connectivity profile.

    Parameters:
    -----------
    eval: complex
        Eigenvalue.
    k: float
        Wavenumber in 1/m.
    tau: float
        Time constant from fit in s.
    W_rate: np.ndarray
        Weights from fit.
    width: np.ndarray
        Spatial widths of boxcar connectivtiy profile in m.

    Returns:
    --------
    xi_eff_r: complex
    """
    omega = complex(0, -eval)
    MH_r = _effective_connectivity_rate(omega, tau, W_rate)
    P_hat = aux_calcs.p_hat_boxcar(k, width)
    xi_eff_r = aux_calcs.determinant_same_rows(MH_r * P_hat)
    return xi_eff_r


def _d_xi_eff_s_d_lambda(eval, k, mu, sigma, tau_m, tau_s, tau_r, Vth_rel,
                         V_0_rel, J, K, dimension, width):
    """
    Computes the derivative of He_lif wrt lambda,
    numerically.
    Requires a spatially organized network with boxcar connectivity profile.

    Parameters:
    -----------
    eval: complex
        Eigenvalue.
    k: float
        Wavenumber in 1/m.
    mu: float
        Mean input in mV.
    sigma: float
        Standard deviation of input in mV.
    tau_m: float
        Membrane time constant in s.
    tau_s: float
        Synaptic time constant in s.
    tau_r: float
        Refractory time in s.
    V_th_rel: float
        Relative threshold potential in mV.
    V_0_rel: float
        Relative reset potential in mV.
    J: np.ndarray
        Weight matrix in mV.
    K: np.ndarray
        Indegree matrix.
    dimension: int
        Dimension of the system / number of populations.
    width: np.ndarray
        Spatial widths of boxcar connectivtiy profile in m.

    Returns:
    --------
    deriv: complex
    """
    def f(x):
        # why is this omega never used?
        omega = complex(0, -eval)
        return _xi_eff_s(eval, k, mu, sigma, tau_m, tau_s, tau_r, Vth_rel,
                         V_0_rel, J, K, dimension, width)

    # TODO: check precision
    deriv = smisc.derivative(func=f, x0=eval, dx=1e-10)
    return deriv


def _d_xi_eff_r_d_lambda(eval, k, tau, W_rate, width):
    """
    Computes the derivative of He_rate wrt lambda,
    analytical expression.
    Requires a spatially organized network with boxcar connectivity profile.

    Parameters:
    -----------
    eval: complex
        Eigenvalue.
    k: float
        Wavenumber in 1/m.
    tau: float
        Time constant from fit in s.
    W_rate: np.ndarray
        Weights from fit.
    width: np.ndarray
        Spatial widths of boxcar connectivtiy profile in m.

    Returns:
    --------
    xi_eff_r: complex
    """
    lp = 1. / (1. + eval * tau)
    deriv = -1. * lp**2 * tau * \
        aux_calcs.determinant(W_rate * aux_calcs.p_hat_boxcar(k, width))
    return deriv


def _solve_chareq_numerically_alpha(lambda_guess, alpha, k, delay, mu, sigma,
                                    tau_m, tau_s, tau_r, V_th_rel, V_0_rel,
                                    J, K, dimension, tau, W_rate, width):
    """
    Uses scipy.optimize.fsolve to solve the characteristic equation.
    Compute the derivative of lambda with respect to alpha.
    Requires a spatially organized network with boxcar connectivity profile.

    Parameters:
    -----------
    lambda_guess: complex
        Guess for eigenvalue.
    alpha: float
        Interpolation parameters.
    k: float
        Wavenumber in 1/m.
    delay: float
        Delay in s.
    mu: float
        Mean input in mV.
    sigma: float
        Standard deviation of input in mV.
    tau_m: float
        Membrane time constant in s.
    tau_s: float
        Synaptic time constant in s.
    tau_r: float
        Refractory time in s.
    V_th_rel: float
        Relative threshold potential in mV.
    V_0_rel: float
        Relative reset potential in mV.
    J: np.ndarray
        Weight matrix in mV.
    K: np.ndarray
        Indegree matrix.
    dimension: int
        Dimension of the system / number of populations.
    tau: float
        Time constant from fit in s.
    W_rate: np.ndarray
        Weights from fit.
    width: np.ndarray
        Spatial widths of boxcar connectivtiy profile in m.

    Returns:
    --------
    lamb: complex

    """
    def fsolve_complex(l_re_im):
        eval = complex(l_re_im[0], l_re_im[1])

        xi_eff_s = _xi_eff_s(eval, k, mu, sigma, tau_m, tau_s, tau_r, V_th_rel,
                             V_0_rel, J, K, dimension, width)
        xi_eff_r = _xi_eff_r(eval, k, tau, W_rate, width)

        xi_eff_alpha = alpha * xi_eff_s + (1. - alpha) * xi_eff_r
        roots = xi_eff_alpha * np.exp(-eval * delay) - 1.

        return [roots.real, roots.imag]

    lambda_guess_list = [np.real(lambda_guess), np.imag(lambda_guess)]
    l_opt = sopt.fsolve(fsolve_complex, lambda_guess_list)
    lamb = complex(l_opt[0], l_opt[1])
    return lamb


@ureg.wraps((None, None, (1 / ureg.mm).units, (1 / ureg.mm).units),
            ((1 / ureg.mm).units, None, ureg.mm))
def xi_of_k(ks, W_rate, width):
    """
    Compute minimum and maximum of spatial profile xi of k
    for linearized rate model.
    Requires a spatially organized network with boxcar connectivity profile.

    Parameters:
    -----------
    ks: Quantity(np.ndarray, '1/mm')
        Range of wavenumbers.
    W_rate: np.ndarray
        Weights from fit.
    width: Quantity(np.ndarray, 'mm')
        Spatial widths of boxcar connectivtiy profile.

    Returns:
    --------
    xi_min: float
    xi_max: float
    k_min: Quantity(float, '1/mm')
    k_max: Quantity(float, '1/mm')
    """
    xis = np.zeros(len(ks))
    for i, k in enumerate(ks):
        P_hat = aux_calcs.p_hat_boxcar(k, width)
        xis[i] = aux_calcs.determinant(W_rate * P_hat)

        xi_min = np.min(xis)
        xi_max = np.max(xis)
        k_min = ks[np.argmin(xis)]
        k_max = ks[np.argmax(xis)]
        if k_min == ks[-1]:
            print('WARNING: k_min==ks[-1] = {}'.format(k_max))
        if k_max == ks[-1]:
            print('WARNING: k_max==ks[-1] = {}'.format(k_max))
    return xi_min, xi_max, k_min, k_max


@ureg.wraps((1 / ureg.s).units,
            (None, (1 / ureg.mm).units, ureg.s, None, ureg.mm, ureg.s))
def solve_chareq_rate_boxcar(branch, k_wavenumber, tau, W_rate, width, delay):
    """
    Solve the characteristic equation for the linearized rate model for
    one branch analytically.
    Requires a spatially organized network with boxcar connectivity profile.

    Parameters:
    -----------
    branch: float
        Branch number.
    k: Quantity(float, '1/mm')
        Wavenumber.
    tau: Quantity(float, 's')
        Time constant from fit.
    W_rate: np.ndarray
        Weights from fit.
    width: Quantity(np.ndarray, 'mm')
        Spatial widths of boxcar connectivtiy profile.
    delay: Quantity(float, 's')
        Delay.

    Returns:
    --------
    eigenval: complex
    """
    eigenval = aux_calcs.solve_chareq_rate_boxcar(branch, k_wavenumber, tau,
                                                  W_rate, width, delay)
    return eigenval
