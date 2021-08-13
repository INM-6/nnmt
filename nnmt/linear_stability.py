import numpy as np
from scipy.special import lambertw


def solve_characteristic_equation_lambertw(
        branch_nr, tau, delay, connectivity):
    """
    Uses the Lambert W function to compute the eigenvalue by solving the
    characteristic equation with delay for a given branch number.

    Parameters
    ----------
    branch_nr: int
        Branch number.
    tau: np.array
        Time constants in s.
    delay: np.array
        Delays in s.
    connectivity:
        Effective connectivity matrix.

    Returns
    -------
    eigenval
    """
    # only scalar or equal value for all populations accepted
    for v in [tau, delay]:
        assert np.isscalar(v) or len(np.unique(v) == 1)
    t, d = np.unique(tau)[0], np.unique(delay)[0]

    c = linalg_max_eigenvalue(connectivity)

    eigenval = -1. / t + 1. / d * \
        lambertw(c * d / t * np.exp(d / t), branch_nr)
    return eigenval


def linalg_max_eigenvalue(matrix):
    """
    Computes the eigenvalue with the largest absolute value of a given matrix.

    Parameters
    ----------
    matrix: np.array
        Matrix to calculate eigenvalues from.

    Returns
    -------
    max_eigval:
    """
    eigvals = np.linalg.eigvals(matrix)
    max_eigval = eigvals[np.argmax(np.abs(eigvals))]
    return max_eigval
