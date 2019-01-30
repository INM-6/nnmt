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
"""
from __future__ import print_function
import ipdb
import warnings
import numpy as np
import pint
import scipy.optimize as sopt
from scipy.special import zetac

from . import ureg
from . import aux_calcs

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
        Effective connectivity matrix.
    j: Quantity(float, 'millivolt')
        Effective connectivity weight.
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
        """ calculate stationary firing rate with given parameters """
        return aux_calcs.nu0_fb433(tau_m, tau_s, tau_r, V_th_rel, V_0_rel, mu,
                                   sigma)

    def get_rate_difference(nu):
        """ calculate difference between new iteration step and previous one """
        ### new mean
        mu = _mean(nu, K, J, j, tau_m, nu_ext, K_ext, g, nu_e_ext, nu_i_ext)

        ### new std
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
        y[1] = y[0] + delta_y*dt
        epsilon = (y[1] - y[0])
        eps = max(np.abs(epsilon))
        y[0] = y[1]

    return y[1]


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
        effective connectivity matrix
    j: Quantity(float, 'millivolt')
        effective connectivity weight
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
    m_ext_add =  tau_m * j * (nu_e_ext - g * nu_i_ext)
    # add them up
    m = m0 + m_ext + m_ext_add

    return m


@ureg.wraps(ureg.mV, (ureg.Hz, None, ureg.mV, ureg.mV, ureg.s, ureg.Hz, None,
                      None, ureg.Hz, ureg.Hz))
def standard_deviation(nu, K, J, j, tau_m, nu_ext, K_ext, g, nu_e_ext, nu_i_ext):
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
        effective connectivity matrix
    j: Quantity(float, 'millivolt')
        effective connectivity weight
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


def _standard_deviation(nu, K, J, j, tau_m, nu_ext, K_ext, g, nu_e_ext, nu_i_ext):
    """ Compute standard_deviation() without quantities. """
    # contribution from within the network to variance
    var0 = tau_m * np.dot(K * J**2, nu)
    # contribution from external sources to variance
    var_ext = tau_m * j**2 * K_ext * nu_ext
    # contribution from additional excitatory and inhibitory Poisson input
    var_ext_add =  tau_m * j**2 * (nu_e_ext + g**2 * nu_i_ext)
    # add them up
    var = var0 + var_ext + var_ext_add
    # standard deviation is square root of variance
    sigma = np.sqrt(var)
    return sigma


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


@ureg.wraps(ureg.Hz/ureg.mV, (ureg.mV, ureg.mV, ureg.s, ureg.s, ureg.s, ureg.mV,
                              ureg.mV, ureg.Hz))
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
    omega: Quantity(float, 'hertz')
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
    dimension: int
        Number of populations.
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
                           for i in range(dimension)]
                          for omega in omegas]

    # convert list of list of quantities to list of quantities containing np.ndarray
    tf_magnitudes = np.array([np.array([tf.magnitude for tf in tf_population])
                     for tf_population in transfer_functions])
    tf_unit = transfer_functions[0][0].units

    return tf_magnitudes * tf_unit

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
        D = np.ones((int(dimension), int(dimension)))
        return D*np.exp(-np.complex(0,omega)*Delay)

    elif delay_dist == 'truncated_gaussian':
        a0 = aux_calcs.Phi(-Delay/Delay_sd+1j*omega*Delay_sd)
        a1 = aux_calcs.Phi(-Delay/Delay_sd)
        b0 = np.exp(-0.5*np.power(Delay_sd*omega,2))
        b1 = np.exp(-np.complex(0,omega)*Delay)
        return (1.0-a0)/(1.0-a1)*b0*b1

    elif delay_dist == 'gaussian':
        b0 = np.exp(-0.5*np.power(Delay_sd*omega,2))
        b1 = np.exp(-np.complex(0,omega)*Delay)
        return b0*b1

def delay_dist_matrix(dimension, Delay, Delay_sd, delay_dist, omegas):
    """ Calculates delay distribution matrices for all omegas. """
    ddms = [delay_dist_matrix_single(dimension, Delay, Delay_sd,
                                             delay_dist, omega)
                           for omega in omegas]

    # convert list of list of quantities to list of quantities containing np.ndarray
    delay_dist_matrices = np.array([ddm.magnitude for ddm in ddms])
    ddm_unit = ddms[0].units

    return delay_dist_matrices * ddm_unit



@ureg.wraps(ureg.dimensionless, (ureg.Hz/ureg.mV, ureg.dimensionless, ureg.mV,
                                 ureg.s, ureg.s, None, ureg.Hz))
def sensitivity_measure(transfer_function, delay_dist_matrix, J, tau_m, tau_s,
                        dimension, omega):
    """
    Calculates sensitivity measure as in Eq. 21 in Bos et al. (2015).

    Parameters:
    -----------
    transfer_function: Quantity(np.ndarray, 'hertz/mV')
        Transfer_function for given frequency omega.
    delay_dist_matrix: Quantity(np.ndarray, 'dimensionless')
        Delay distribution matrix at given frequency.
    J: Quantity(np.ndarray, 'millivolt')
        Effective connectivity matrix.
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

    if omega < 0:
        transfer_function = np.conjugate(transfer_function)
    H = tau_m * transfer_function.T / complex(1, omega*tau_s)
    H = np.hstack([H for i in range(dimension)])
    H = np.transpose(H.reshape(dimension,dimension))
    MH = H*J*delay_dist_matrix

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

@ureg.wraps(ureg.Hz, (ureg.s, ureg.s, None, ureg.mV, None, ureg.dimensionless, None,
                   ureg.Hz, ureg.Hz/ureg.mV, ureg.Hz))
def power_spectra(tau_m, tau_s, dimension, J, K, delay_dist_matrix, N,
                  firing_rates, transfer_function, omegas):
    """
    Calculates vector of power spectra for all populations at given frequencies.

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
        Effective connectivity matrix.
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

        if omega < 0:
            transfer_function = np.conjugate(transfer_function)
        H = tau_m * transfer_function.T / complex(1, omega*tau_s)
        H = np.hstack([H for i in range(dimension)])
        H = np.transpose(H.reshape(dimension,dimension))
        MH = H*J*K*delay_dist_matrix

        Q = np.linalg.inv(np.identity(dimension)-MH)
        D = (np.diag(np.ones(dimension)) * firing_rates / N)
        C = np.dot(Q, np.dot(D, np.transpose(np.conjugate(Q))))
        spec = np.absolute(np.diag(C))
        return spec

    power = np.array([power_spectra_single_freq(tau_m, tau_s, transfer_function[i],
                                       dimension, J, K, delay_dist_matrix[i],
                                       firing_rates, N, omega)
             for i,omega in enumerate(omegas)])


    return np.transpose(power)



@ureg.wraps(ureg.dimensionless, (ureg.s, ureg.s, ureg.Hz/ureg.mV, None,
                                 ureg.dimensionless, ureg.mV, ureg.Hz, None,
                                 None))
def eigen_spectra(tau_m, tau_s, transfer_function, dimension,
                  delay_dist_matrix, J, omegas, quantity, matrix):
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
        Effective connectivity matrix.
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
                                  delay_dist_matrix, J, omega, matrix):

        if omega < 0:
            transfer_function = np.conjugate(transfer_function)
        H = tau_m * transfer_function.T / complex(1, omega*tau_s)
        H = np.hstack([H for i in range(dimension)])
        H = np.transpose(H.reshape(dimension,dimension))
        MH = H*J*delay_dist_matrix

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
        eig = [eigen_spectra_single_freq(tau_m, tau_s, transfer_function[i], dimension,
                             delay_dist_matrix[i], J, omega, matrix)[0]
               for i,omega in enumerate(omegas)]
    elif quantity == 'reigvecs':
        eig = [eigen_spectra_single_freq(tau_m, tau_s, transfer_function[i], dimension,
                                         delay_dist_matrix[i], J, omega, matrix)[1]
                           for i,omega in enumerate(omegas)]
    elif quantity == 'leigvecs':
        eig = [eigen_spectra_single_freq(tau_m, tau_s, transfer_function[i], dimension,
                                        delay_dist_matrix[i], J, omega, matrix)[2]
                          for i,omega in enumerate(omegas)]

    return eig


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
        Effective connectivity matrix.
    j: Quantity(float, 'millivolt')
        Effective connectivity weight.
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
        target_rates[i] = aux_calcs.nu0_fb433(tau_m, tau_s, tau_r,
                                              V_th_rel, V_0_rel,
                                              mu_set[i], sigma_set[i])

    # additional external rates set to 0 for local-only contributions
    mu_loc =_mean(nu=target_rates, K=K, J=J, j=j, tau_m=tau_m,
                  nu_ext=nu_ext, K_ext=K_ext,
                  g=g, nu_e_ext=0., nu_i_ext=0.)
    sigma_loc = _standard_deviation(nu=target_rates, K=K, J=J, j=j, tau_m=tau_m,
                                    nu_ext=nu_ext, K_ext=K_ext,
                                    g=g, nu_e_ext=0., nu_i_ext=0.)

    mu_temp = (mu_set - mu_loc) / (tau_m * j)
    sigma_temp_2 = (sigma_set**2 - sigma_loc**2) / (tau_m * j**2)

    nu_e_ext = (sigma_temp_2 + g * mu_temp) / (1. + g)
    nu_i_ext = (sigma_temp_2 - mu_temp) / (g * (1. + g))

    if np.any(np.array([nu_e_ext, nu_i_ext]) < 0):
        warn = 'Negative rate detected:\n\tnu_e_ext=' + str(nu_e_ext) + '\n\tnu_i_ext=' + str(nu_i_ext)
        warnings.warn(warn)

    return nu_e_ext, nu_i_ext


@ureg.wraps((ureg.s, None, None, ureg.Hz/ureg.mV),
            (ureg.Hz/ureg.mV, ureg.Hz, ureg.s, ureg.mV, None))
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
        Effective connectivity matrix.
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

    return tau_rate, W_rate, W_rate_sim, fit_tf


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
        Time constant from fit in s.
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

    fit_tf = np.zeros(np.shape(transfer_function), dtype=np.complex_)
    dim = np.shape(transfer_function)[1]
    tau_rate = np.zeros(dim)
    h0 = np.zeros(dim)
    err_tau = np.zeros(dim)
    err_h0 = np.zeros(dim)

    bounds = [[0., -np.inf],[np.inf, np.inf]]
    for i in np.arange(np.shape(fit_tf)[1]):
        fitParams, fitCovariances = sopt.curve_fit(func_abs,
                                                   omegas,
                                                   np.abs(transfer_function[:,i]),
                                                   bounds=bounds)
        tau_rate[i] = fitParams[0]
        h0[i] = fitParams[1]
        fit_tf[:,i] = func(omegas, tau_rate[i], h0[i])

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
        Relative error on fitted tau for each combination of mean and std of input.
    errs_h0: np.ndarray
        Relative error on fitted h0 for each combination of mean and std of input.
    """
    dims = (len(mean_inputs), len(std_inputs))
    errs_tau = np.zeros(dims)
    errs_h0 = np.zeros(dims)

    for i,mu in enumerate(mean_inputs):
        for j,sigma in enumerate(std_inputs):
            if i==0 and j==0: # get unit, same for all
                unit = transfer_function_1p_shift(mu, sigma, tau_m,tau_s, tau_r,
                                                  V_th_rel, V_0_rel,
                                                  omegas[0]).units

            transfer_function = [[transfer_function_1p_shift(mu, sigma, tau_m,
                                                            tau_s, tau_r, V_th_rel,
                                                            V_0_rel, omega).magnitude]
                                    for omega in omegas] * unit

            fit_tf, tau_rate, h0, err_tau, err_h0 = \
                _fit_transfer_function( \
                    transfer_function.to(ureg.Hz / ureg.mV).magnitude,
                    omegas.to(ureg.Hz).magnitude)

            errs_tau[i,j] = err_tau[0]
            errs_h0[i,j] = err_h0[0]
    return errs_tau, errs_h0


@ureg.wraps(None, (ureg.s, ureg.s, ureg.s, ureg.mV, ureg.mV, ureg.mV, ureg.mV, ureg.mV))
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
        Effective connectivity matrix.
    j: Quantity(float, 'millivolt')
        Effective connectivity weight.
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
            w_ecs[post][pre] = aux_calcs.d_nu_d_nu_in_fb(
                tau_m, tau_s, tau_r, V_th_rel, V_0_rel, J[post][pre],
                mean_input[pre], std_input[pre])[1] # linear (mu) contribution
    return w_ecs
