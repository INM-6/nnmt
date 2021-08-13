import numpy as np


def spatial_profile_boxcar(k, width):
    """
    Fourier transform of boxcar connectivity kernel at wave number k.

    Parameters:
    -----------
    k: float
        Wavenumber in 1/m.
    width: float or np.ndarray
        Width(s) of boxcar kernel(s) in m.

    Returns:
    --------
    ft: float
    """
    if k == 0:
        ft = 1.
    else:
        ft = np.sin(k * width) / (k * width)
    return ft
