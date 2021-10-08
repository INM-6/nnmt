"""
Defines the microcircuit model, a multilayer model of a cortical column.
"""

import numpy as np

from .network import Network
from .. import ureg


class Microcircuit(Network):
    """
    The Potjans and Diesmann microcircuit model.

    See :cite:t:`potjans2014` for details regarding the model. In short, it is
    a four-layer (2/3, 4, 5, 6) network model with a population of excitatory
    (E) and inhibitory (I) neurons of leaky integrate-and-fire neurons with
    exponential synapses in each layer. The inhibitory synaptic weights are `g`
    times as strong as the excitatory synaptic weights. The weights between all
    populations are equally strong, except for layer 4E to layer 2/3E, where
    the excitatory weights are twice as strong.

    Given the parameter yaml files, the network model calculates the dependend
    parameters. It converts the weights from pA to mV, calculates the weight
    matrix, calculates relative thresholds, and the analysis frequencies.

    The NNMT repository contains an :ref:`example <sec_examples>` providing the
    yaml parameter files with all the parameters that need to be defined to use
    this model.

    Parameters
    ----------
    network_params : [str | dict]
        Network parameters yaml file name or dictionary including:

        - `C` : float
            Membrane capacitance in pF.
        - `K_ext` : np.array
            Number of external in-degrees.
        - `V_th_abs` : [float | np.array]
            Absolute threshold potential in mV.
        - `V_0_abs` : [float | np.array]
            Absolute reset potential in mV.
        - `d_e` : float
            Mean delay of excitatory connections in ms.
        - `d_e_sd` : float
            Standard deviation of delay of excitatory connections in ms.
        - `d_i` :  float
            Mean delay of inhibitory connections in ms.
        - `d_i_sd`
            Standard deviation of delay of inhibitory connections in ms.
        - `g` : float
            Ratio of inhibitory to excitatory synaptic weights.
        - `populations` : list of strings
            Names of different populations.
        - `tau_s` : float
            Synaptic time constant in ms.
        - `w` : float
            Amplitude of excitatory post synaptic current in pA.
        - `w_ext`: float
            Amplitude of external excitatory post synaptic current in pA.

    analysis_params : [str | dict]
        Analysis parameters yaml file name or dictionary including:

        - `df` : float
            Step size between two analysis frequencies.
        - `f_min` : float
            Minimal analysis frequency.
        - `f_max` : float
            Maximal analysis frequency.
        - `dk` : float
            Step size between two analysis wavenumber.
        - `k_min` : float
            Minimum analysis wavenumber.
        - `k_max`
            Maximum analysis wavenumber.

    See Also
    --------
    nnmt.models.Network : Parent class defining all arguments, attributes, and
                          methods.

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
        Calculate all network parameters derived from parameters in yaml file.

        Calculates the number of populations, the relative potentials, converts
        the weights from pA to mV, constructs the weight matrix, and the delay
        matrix.

        Returns:
        --------
        dict
            Dictionary containing all derived network parameters.
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

        # mean delay matrix
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

        Calculates the angular analysis frequencies, and optionally the
        wavenumbers for a spatial analysis.

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

        try:
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
        except KeyError:
            pass

        return derived_params
