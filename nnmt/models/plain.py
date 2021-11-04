"""
Defines Plain network model. A plain network without any assumptions.
"""

from .network import Network


class Plain(Network):
    """
    Plain network model that does not make any assumptions.

    Network and analysis parameters are read in, converted to SI units, and
    the units are stripped off.

    See Also
    --------
    nnmt.models.Network : Parent class defining all arguments, attributes, and
                          methods.
    """

    def __init__(self, network_params=None, analysis_params=None, file=None):

        super().__init__(network_params, analysis_params, file)

        self._convert_param_dicts_to_base_units_and_strip_units()

    def _instantiate(self, new_network_params, new_analysis_params):
        return Plain._instantiate(new_network_params, new_analysis_params)
