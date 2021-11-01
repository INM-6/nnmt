"""
Functions for spatially structured networks.

Parameter Functions
*******************

.. autosummary::
    :toctree: _toctree/

_spatial_profile_boxcar

"""

import numpy as np


def _spatial_profile_boxcar(k, width):
    """
    Fourier transform of boxcar connectivity kernel at given wave number.

    Parameters
    ----------
    k : float
        Wavenumber in 1/m.
    width : float or np.ndarray
        Width(s) of boxcar kernel(s) in m.

    Returns
    -------
    ft : float
        Fourier transform of spatial profile.
    """
    if k == 0:
        ft = 1.
    else:
        ft = np.sin(k * width) / (k * width)
    return ft
