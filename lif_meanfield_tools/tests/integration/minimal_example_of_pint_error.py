# use lmt_clean

import sys
sys.path.append('/home/essink/working_environment/lif_meanfield_tools')
import lif_meanfield_tools as lmt
from lif_meanfield_tools.__init__ import ureg

# Problem 1
transfer_function = lmt.meanfield_calcs.transfer_function_1p_shift(
    1 * ureg.mV, 1 * ureg.mV, 1 * ureg.s, 1 * ureg.s, 1 * ureg.s, 1 * ureg.mV,
    1 * ureg.mV, 1 * ureg.Hz)
print(transfer_function)

"""
Hot Guess: I have checked that the two unit registries (in this script, and
in the use lmt.meanfield_calcs()) have different ID's. Thus most most likely
the wrapper which invoques conversion can't recognize the correct unit.
"""


# Problem 2
"""
In some @wraps-decorator 1/ureg.mm was used, which is a Quantity object,
however @wraps in the newest pint-version installable via conda seems
just to accept str and units.
"""
type(ureg.Hz)
type((1/ureg.Hz).units)

# one hotfix could be
(1/ureg.Hz).units
