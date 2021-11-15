"""
Functions for linear stability analysis.

Parameter Functions
*******************

.. autosummary::
    :toctree: _toctree/linear_stability/

    _solve_chareq_lambertw_constant_delay

"""

import numpy as np
from scipy.special import lambertw


def _solve_chareq_lambertw_constant_delay(
        branch_nr, tau, delay, connectivity):
    """
    Calcs the most critical eigenvalue for neural networks with constant delay.

    Uses the Lambert W function to solve the characteristic equation for the
    respective system. This function can be applied to networks without (Eq. 1
    in :cite:t:`helias2013`) and with (Eq. 1 in :cite:t:`senk2020`) a spatially
    structured connectivity. It computes the most critical temporal eigenvalue
    given by Eq. 8 in :cite:t:`helias2013` or Eq. 7 in :cite:t:`senk2020`
    respectively for any given branch number.

    Parameters
    ----------
    branch_nr : int
        Branch number.
    tau : np.array
        Time constants in s.
    delay : np.array
        Delays in s.
    connectivity : np.array
        Matrix defining the connectivity. For non-spatial networks, this is
        just the weight matrix. For spatial networks, this is an effective
        connectivity matrix; each element is the weight multiplied with the
        Fourier transform of the spatial profile at the wave number k for which
        the characteristic equation is to be evaluated.

    Returns
    -------
    eigenval : np.complex
        Temporal eigenvalue solving the characteristic equation.
    """
    # only scalar or equal value for all populations accepted
    for v in [tau, delay]:
        assert (np.isscalar(v) or len(np.unique(v)) == 1)
    t, d = np.unique(tau)[0], np.unique(delay)[0]

    # eigenvalue of connectivity matrix with largest absolute value.
    # (an example for these eigenvalues is given in Senk et al. (2020), Eq. 5)
    cs = np.linalg.eigvals(connectivity)
    c = cs[np.argmax(np.abs(cs))]

    eigenval = (-1. / t + 1. / d
                * lambertw(c * d / t * np.exp(d / t), branch_nr))
    return eigenval
