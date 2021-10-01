"""
Module that defines the a homogeneous binomial network model.

"""

import numpy as np

from .network import Network


class HomogeneousBinomial(Network):
    """
    A very basic homogeneous random binomial network.

    See Also
    --------
    nnmt.models.Network : Parent class

    """

    def __init__(self, network_params=None, analysis_params=None, file=None):

        super().__init__(network_params, analysis_params, file)

        self.calc_connectivity_weight()

        self._convert_param_dicts_to_base_units_and_strip_units()

    def _instantiate(self, new_network_params, new_analysis_params):
        return HomogeneousBinomial(new_network_params, new_analysis_params)

    def calc_connectivity_weight(self):
        r = self.network_params['r']
        N = self.network_params['N']
        p = self.network_params['p']
        self.network_params['w'] = np.sqrt(r**2 / (N * p * (1 - p)))