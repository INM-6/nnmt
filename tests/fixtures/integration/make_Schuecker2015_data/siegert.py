"""siegert.py: Function calculating the firing rates of leaky
integrate-and-fire neurons given their parameter and mean and variance
of the input. Rates rates for delta shaped PSCs after Brunel & Hakim 1999.
Rate of neuron with synaptic filtering with time constant tau_s after
Fourcoud & Brunel 2002.

Authors: Moritz Helias, Jannis Schuecker, Hannah Bos
"""

from scipy.integrate import quad
from scipy.special import erf
from scipy.special import zetac
import numpy as np
import math

"""
Variables used in this module:
tau_m: membrane time constant
tau_r: refractory time constant
V_th: threshold
V_r: reset potential
mu: mean input
sigma: std of equivalent GWN input
"""

def nu_0(tau_m, tau_r, V_th, V_r, mu, sigma):
    """ Calculates stationary firing rates for delta shaped PSCs."""

    if mu <= V_th + (0.95 * abs(V_th) - abs(V_th)):
        return siegert1(tau_m, tau_r, V_th, V_r, mu, sigma)
    else:
        return siegert2(tau_m, tau_r, V_th, V_r, mu, sigma)

# stationary firing rate of neuron with synaptic low-pass filter
# of time constant tau_s driven by Gaussian noise with mean mu and
# standard deviation sigma, from Fourcaud & Brunel 2002
def nu0_fb433(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma):
    """Calculates stationary firing rates for exponential PSCs using
    expression with taylor expansion in k = sqrt(tau_s/tau_m) (Eq. 433
    in Fourcoud & Brunel 2002)
    """

    alpha = np.sqrt(2.) * abs(zetac(0.5) + 1)
    x_th = np.sqrt(2.) * (V_th - mu) / sigma
    x_r = np.sqrt(2.) * (V_r - mu) / sigma

    # preventing overflow in np.exponent in Phi(s)
    if x_th > 20.0 / np.sqrt(2.):
        result = nu_0(tau_m, tau_r, V_th, V_r, mu, sigma)
    else:
        r = nu_0(tau_m, tau_r, V_th, V_r, mu, sigma)
        dPhi = Phi(x_th) - Phi(x_r)
        result = r - np.sqrt(tau_s / tau_m) * alpha / \
            (tau_m * np.sqrt(2)) * dPhi * (r * tau_m)**2
    if math.isnan(result):
        print mu, sigma, x_th, x_r
    return result


def nu0_fb(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma):
    """
    Calculates stationary firing rates for filtered synapses based on
    Fourcaud & Brunel 2002.

    Parameters:
    -----------
    tau_m: float
        Membrane time constant in seconds.
    tau_s: float
        Synaptic time constant in seconds.
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

    alpha = np.sqrt(2)*abs(zetac(0.5)+1)
    # effective threshold
    V_th1 = V_th + sigma*alpha/2.*np.sqrt(tau_s/tau_m)
    # effective reset
    V_r1 = V_r + sigma*alpha/2.*np.sqrt(tau_s/tau_m)
    # use standard Siegert with modified threshold and reset
    return nu_0(tau_m, tau_r, V_th1, V_r1, mu, sigma)


def Phi(s):
    return np.sqrt(np.pi / 2.) * (np.exp(s**2 / 2.) * (1 + erf(s / np.sqrt(2))))

# previously defined Phi_prime_mu(s), but called with (s, sigma)
# which is obviously needed
def Phi_prime_mu(s, sigma):
    return -np.sqrt(np.pi) / sigma * (s * np.exp(s**2 / 2.) * (1 + erf(s / np.sqrt(2)))
                                      + np.sqrt(2) / np.sqrt(np.pi))

def siegert1(tau_m, tau_r, V_th, V_r, mu, sigma):
    # for mu < V_th
    y_th = (V_th - mu) / sigma
    y_r = (V_r - mu) / sigma

    def integrand(u):
        if u == 0:
            return np.exp(-y_th**2) * 2 * (y_th - y_r)
        else:
            return np.exp(-(u - y_th)**2) * (1.0 - np.exp(2 * (y_r - y_th) * u)) / u

    lower_bound = y_th
    err_dn = 1.0
    while err_dn > 1e-12 and lower_bound > 1e-16:
        err_dn = integrand(lower_bound)
        if err_dn > 1e-12:
            lower_bound /= 2

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

def siegert2(tau_m, tau_r, V_th, V_r, mu, sigma):
    # for mu > V_th
    y_th = (V_th - mu) / sigma
    y_r = (V_r - mu) / sigma

    def integrand(u):
        if u == 0:
            return 2 * (y_th - y_r)
        else:
            return (np.exp(2 * y_th * u - u**2) - np.exp(2 * y_r * u - u**2)) / u

    upper_bound = 1.0
    err = 1.0
    while err > 1e-12:
        err = integrand(upper_bound)
        upper_bound *= 2

    return 1.0 / (tau_r + quad(integrand, 0.0, upper_bound)[0] * tau_m)

def d_nu_d_mu_fb433(tau_m, tau_s, tau_r, V_th, V_r, mu, sigma):
    alpha = np.sqrt(2) * abs(zetac(0.5) + 1)
    x_th = np.sqrt(2) * (V_th - mu) / sigma
    x_r = np.sqrt(2) * (V_r - mu) / sigma
    integral = 1. / (nu_0(tau_m, tau_r, V_th, V_r, mu, sigma) * tau_m)
    prefactor = np.sqrt(tau_s / tau_m) * alpha / (tau_m * np.sqrt(2))
    dnudmu = d_nu_d_mu(tau_m, tau_r, V_th, V_r, mu, sigma)
    dPhi_prime = Phi_prime_mu(x_th, sigma) - Phi_prime_mu(x_r, sigma)
    dPhi = Phi(x_th) - Phi(x_r)
    phi = dPhi_prime * integral + (2 * np.sqrt(2) / sigma) * dPhi**2
    return dnudmu - prefactor * phi / integral**3

def d_nu_d_mu(tau_m, tau_r, V_th, V_r, mu, sigma):
    y_th = (V_th - mu)/sigma
    y_r = (V_r - mu)/sigma
    nu0 = nu_0(tau_m, tau_r, V_th, V_r, mu, sigma)
    return np.sqrt(np.pi) * tau_m * nu0**2 / sigma * (np.exp(y_th**2) * (1 + erf(y_th)) - np.exp(y_r**2) * (1 + erf(y_r)))
