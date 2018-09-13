import numpy as np
from my_io import ureg
import pint

def firing_rates():
    return np.arange(10) * ureg.meter

@ureg.wraps(ureg.meter, ureg.meter)
def mean(firing_rates):
    return firing_rates * 2

@ureg.wraps(ureg.meter**2, ureg.meter)
def variance(firing_rates):
    return np.power(firing_rates,  2)
