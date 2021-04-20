"""circuit.py: Main class providing functions to calculate the stationary
and dynamical properties of a given circuit.

Authors: Hannah Bos, Jannis Schuecker
"""

import numpy as np
from setup import Setup
from analytics import Analytics


class Circuit(object):
    """Provides functions to calculate the stationary and dynamical 
    properties of a given circuit.

    Arguments:
    label: string specifying circuit, options: 'microcircuit'

    Keyword Arguments:
    params: dictionary specifying parameter of the circuit, default 
            parameter given in params_circuit.py will be overwritten
    analysis_type: string specifying level of analysis that is requested
                   default: 'dynamical'
                   options: 
                   - None: only circuit and default analysis parameter 
                     are set
                   - 'stationary': circuit and default analysis parameter
                      are set, mean and variance of input to each 
                      populations as well as firing rates are calculated
                   - 'dynamical': circuit and default analysis parameter
                      are set, mean and variance of input to each 
                      populations as well as firing rates are calculated,
                      variables for calculation of spectra are calculated 
                      including the transfer function for all populations
    fmin: minimal frequency in Hz, default: 0.1 Hz
    fmax: maximal frequency in Hz, default: 150 Hz
    df: frequency spacing in Hz, default: 1.0/(2*np.pi) Hz
    to_file: boolean specifying whether firing rates and transfer 
             functions are written to file, default: True
    from_file: boolean specifying whether firing rates and transfer 
               functions are read from file, default: True
               if set to True and file is not found firing rates and
               transfer function are calculated
    """
    def __init__(self, label, params={}, **kwargs):
        self.label = label
        self.setup = Setup()
        self.ana = Analytics()
        if 'analysis_type' in kwargs:
            self.analysis_type = kwargs['analysis_type']
        else: 
            self.analysis_type = 'dynamical'
        # set default analysis and circuit parameter
        self._set_up_circuit(params, kwargs)
        # set parameter derived from analysis and circuit parameter
        new_vars = self.setup.get_params_for_analysis(self)
        new_vars['label'] = self.label
        self._set_class_variables(new_vars)
        # set variables which require calculation in analytics class
        self._calc_variables()

    # updates variables of Circuit() and Analysis() classes, new variables
    # are specified in the dictionary new_vars
    def _set_class_variables(self, new_vars):
        for key, value in new_vars.items():
            setattr(self, key, value)
        if 'params' in new_vars:
            for key, value in new_vars['params'].items():
                setattr(self, key, value)
        self.ana.update_variables(new_vars)

    # updates class variables of variables of Circuit() and Analysis()
    # such that default analysis and circuit parameters are known
    def _set_up_circuit(self, params, args):
        # set default analysis parameter
        new_vars = self.setup.get_default_params(args)
        self._set_class_variables(new_vars)
        # set circuit parameter
        new_vars = self.setup.get_circuit_params(self, params)
        self._set_class_variables(new_vars)

    # quantities required for stationary analysis are calculated
    def _set_up_for_stationary_analysis(self):
        new_vars = self.setup.get_working_point(self)
        self._set_class_variables(new_vars)

    # quantities required for dynamical analysis are calculated
    def _set_up_for_dynamical_analysis(self):
        new_vars = self.setup.get_params_for_power_spectrum(self)
        self._set_class_variables(new_vars)

    # calculates quantities needed for analysis specified by analysis_type
    def _calc_variables(self):
        if self.analysis_type == 'dynamical':
            self._set_up_for_stationary_analysis()
            self._set_up_for_dynamical_analysis()
        elif self.analysis_type == 'stationary':
            self._set_up_for_stationary_analysis()

    def alter_default_params(self, params):
        """Parameter specified in dictionary params are changed. 
        Changeable parameters are default analysis and circuit parameter,
        as well as label and analysis_type.

        Arguments:
        params: dictionary, specifying new parameters 
        """
        self.params.update(params)
        new_vars = self.setup.get_altered_circuit_params(self, self.label)
        self._set_class_variables(new_vars)
        new_vars = self.setup.get_params_for_analysis(self)
        self._set_class_variables(new_vars)
        self._calc_variables()

    def create_power_spectra(self):
        """Returns frequencies and power spectra. 
        See: Eq. 9 in Bos et al. (2015)

        Output:
        freqs: vector of frequencies in Hz
        power: power spectra for all populations, 
               dimension len(self.populations) x len(freqs)
        """
        freqs, power = self.ana.power_spectrum()
        return freqs, power

    def create_power_spectra_approx(self):
        """Returns frequencies and power spectra approximated by 
        dominant eigenmode. 
        See: Eq. 15 in Bos et al. (2015)

        Output:
        freqs: vector of frequencies in Hz
        power: power spectra for all populations, 
               dimension len(self.populations) x len(freqs)
        """
        freqs, power = self.ana.power_spectrum_approx()
        return freqs, power
        
    def create_eigenvalue_spectra(self, matrix):
        """Returns frequencies and eigenvalue spectrum of matrix.

        Arguments:
        matrix: string specifying the matrix, options are the effective
                connectivity matrix ('MH'), the propagator ('prop') and 
                the inverse of the propagator ('prop_inv')

        Output:
        freqs: vector of frequencies in Hz
        power: power spectrum for all populations, 
               dimension len(self.populations) x len(freqs)
        """
        freqs, eigenvalues = self.ana.eigenvalue_spectrum(matrix)
        return freqs, eigenvalues

    def reduce_connectivity(self, M_red):
        """Connectivity (indegree matrix) is reduced, while the working 
        point is held constant. 

        Arguments:
        M_red: matrix, with each element specifying how the corresponding
               connection is altered, e.g the in-degree from population
               j to population i is reduced by 30% with M_red[i][j]=0.7
        """
        M_original = self.M_full[:]
        if M_red.shape != M_original.shape:
            raise RuntimeError('Dimension of mask matrix has to be the '
                               + 'same as the original indegree matrix.')
        self.M = M_original*M_red
        self.ana.update_variables({'M': self.M})

    def restore_full_connectivity(self):
        '''Restore connectivity to full connectivity.'''
        self.M = self.M_full
        self.ana.update_variables({'M': self.M})

    def get_effective_connectivity(self, freq): 
        """Returns effective connectivity matrix.

        Arguments:
        freq: frequency in Hz
        """
        return self.ana.create_MH(2*np.pi*freq)

    def get_sensitivity_measure(self, freq, index=None):
        """Returns sensitivity measure.
        see: Eq. 21 in Bos et al. (2015)

        Arguments:
        freq: frequency in Hz

        Keyword arguments:
        index: specifies index of eigenmode, default: None
               if set to None the dominant eigenmode is assumed
        """
        MH  = self.get_effective_connectivity(freq)
        e, U = np.linalg.eig(MH)    
        U_inv = np.linalg.inv(U)
        if index is None:
            # find eigenvalue closest to one
            index = np.argmin(np.abs(e-1))
        T = np.outer(U_inv[index],U[:,index])
        T /= np.dot(U_inv[index],U[:,index])
        T *= MH
        return T

        
