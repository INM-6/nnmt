import numpy as np
from my_io import ureg
import pint

def firing_rates():
    return np.arange(8) * ureg.Hz

# @ureg.wraps(ureg., ureg.Hz)
def mean(nu, K, W, J, K_ext, nu_ext):
    '''Returns vector of mean inputs to populations depending on
    the firing rates nu of the populations.

    Units as in Fourcoud & Brunel 2002, where current and potential
    have the same units, unit of output array: pA*Hz
    '''

    # need to define this function, because np.dot is not compatible with pint
    # quantities
    @ureg.wraps(ureg.mV * ureg.Hz, (ureg.mV, ureg.Hz))
    def dot(matrix, vector):
        return np.dot(matrix, vector)

    # contribution from within the network
    m0 = dot(K*W, nu)
    # contribution from external sources
    m_ext = J*K_ext*nu_ext
    m = m0+m_ext
    return m
