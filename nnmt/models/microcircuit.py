"""
Module that defines the microcircuit.

"""

import numpy as np

from .network import Network
from .. import ureg


class Microcircuit(Network):
    """
    The Potjans and Diesmann microcircuit model.

    See :cite:t:`potjans2014` for details.

    See Also
    --------
    nnmt.models.Network : Parent class

    """

    def __init__(self, network_params=None, analysis_params=None, file=None):

        super().__init__(network_params, analysis_params, file)

        self.network_params['label'] = 'microcircuit'
        derived_network_params = (
            self._calculate_dependent_network_parameters())
        self.network_params.update(derived_network_params)

        # calculate dependend analysis parameters
        if analysis_params is not None:
            derived_analysis_params = (
                self._calculate_dependent_analysis_parameters())
            self.analysis_params.update(derived_analysis_params)

        self._convert_param_dicts_to_base_units_and_strip_units()

    def _instantiate(self, new_network_params, new_analysis_params):
        return Microcircuit(new_network_params, new_analysis_params)

    def _calculate_dependent_network_parameters(self):
        """
        Calculate all network parameters derived from parameters in yaml file

        Returns:
        --------
        dict
            dictionary containing all derived network parameters
        """
        derived_params = {}

        # calculate dimension of system
        dim = len(self.network_params['populations'])
        derived_params['dimension'] = dim

        # reset reference potential to 0
        derived_params['V_0_rel'] = 0 * ureg.mV
        derived_params['V_th_rel'] = (self.network_params['V_th_abs']
                                      - self.network_params['V_0_abs'])

        # convert weights in pA (current) to weights in mV (voltage)
        tau_s_div_C = self.network_params['tau_s'] / self.network_params['C']
        derived_params['j'] = (tau_s_div_C
                               * self.network_params['w'])
        try:
            derived_params['j'].ito(ureg.mV)
        except AttributeError:
            pass

        # weight matrix in pA (current)
        W = np.ones((dim, dim)) * self.network_params['w']
        W[1:dim:2] *= -self.network_params['g']
        W = np.transpose(W)
        derived_params['W'] = W

        # weight matrix in mV (voltage)
        derived_params['J'] = (tau_s_div_C * derived_params['W'])
        try:
            derived_params['J'].ito(ureg.mV)
        except AttributeError:
            pass

        # delay matrix
        D = np.ones((dim, dim)) * self.network_params['d_e']
        D[1:dim:2] = np.ones(dim) * self.network_params['d_i']
        D = np.transpose(D)
        derived_params['Delay'] = D

        # delay standard deviation matrix
        D = np.ones((dim, dim)) * self.network_params['d_e_sd']
        D[1:dim:2] = np.ones(dim) * self.network_params['d_i_sd']
        D = np.transpose(D)
        derived_params['Delay_sd'] = D

        # larger weight for L4E->L23E connections
        derived_params['W'][0][2] *= 2.0
        derived_params['J'][0][2] *= 2.0

        derived_params['J_ext'] = (
            tau_s_div_C * np.ones(self.network_params['K_ext'].shape)
            * self.network_params['w_ext'])
        try:
            derived_params['J_ext'].ito(ureg.mV)
        except AttributeError:
            pass

        return derived_params

    def _calculate_dependent_analysis_parameters(self):
        """
        Calculate all analysis parameters derived from parameters in yaml file

        Returns:
        --------
        dict
            dictionary containing derived parameters
        """
        derived_params = {}

        # convert regular to angular frequencies
        w_min = 2 * np.pi * self.analysis_params['f_min']
        w_max = 2 * np.pi * self.analysis_params['f_max']
        dw = 2 * np.pi * self.analysis_params['df']

        try:
            w_min = w_min.magnitude
            w_max = w_max.magnitude
            dw = dw.magnitude
        except AttributeError:
            pass

        # enable usage of quantities
        def calc_evaluated_omegas(w_min, w_max, dw):
            """ Calculates omegas at which functions are to be evaluated """
            return np.arange(w_min, w_max, dw)

        derived_params['omegas'] = calc_evaluated_omegas(w_min, w_max, dw)

        try:
            w_min = w_min.magnitude
            w_max = w_max.magnitude
            dw = dw.magnitude
        except AttributeError:
            pass

        def calc_evaluated_wavenumbers(k_min, k_max, dk):
            return np.arange(k_min, k_max, dk)

        k_min = self.analysis_params['k_min']
        k_max = self.analysis_params['k_max']
        dk = self.analysis_params['dk']

        try:
            k_min = k_min.to_base_units().magnitude
            k_max = k_max.to_base_units().magnitude
            dk = dk.to_base_units().magnitude
        except AttributeError:
            pass

        derived_params['k_wavenumbers'] = (
            calc_evaluated_wavenumbers(k_min, k_max, dk))

        return derived_params
