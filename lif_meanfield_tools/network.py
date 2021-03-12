"""
Main class providing functions to calculate the stationary and
dynamical properties of a given network.

Authors: 2016: Hannah Bos, Jannis Schuecker
         2018: Moritz Layer, Johanna Senk, Karolina Korvasova

Classes:
--------
Network
    Object for storing network and analysis parameters. It has lots of
    different methods for calculating properties of the network.

Network Methods:
----------------
__init__
save
show
change_parameters
firing_rates
mean
standard_deviation
working_point
delay_dist_matrix
delay_dist_matrix_multi
delay_dist_matrix_single
transfer_function
transfer_function_multi
transfer_function_single
sensitivity_measure
power_spectra
eigenvalue_spectra
r_eigenvec_spectra
l_eigenvec_spectra
additional_rates_for_fixed_input
fit_transfer_function
scan_fit_transfer_function_mean_std_input
linear_interpolation_alpha
_calculate_dependent_network_parameters
_calculate_dependent_analysis_parameters
_check_and_store
"""

from __future__ import print_function
import numpy as np
from decorator import decorator
import hashlib

from . import ureg
from . import input_output as io
from . import meanfield_calcs


class Network(object):
    """
    Network with given parameters. The class provides methods for calculating
    stationary and dynamical properties of the defined network.

    Parameters:
    -----------
    network_params: str
        specifies path to yaml file containing network parameters
        if None only new_network_params are used
    analysis_params: str
        specifies path to yaml file containing analysis parameters
        if None only new_analysis_params are used
    new_network_params: dict
        dictionary specifying network parameters from yaml file that should be
        overwritten. Format:
        {'<param1>:{'val':<value1>, 'unit':<unit1>},...}
    new_analysis_params: dict
        dictionary specifying analysis parameters from yaml file that should be
        overwritten. Format:
        {'<param1>:{'val':<value1>, 'unit':<unit1>},...}
    derive_params: bool
        whether parameters shall be derived from existing ones
        can be false if a complete set of network parameters is given
    file: str
        file name of h5 file from which network can be loaded
    """

    def __init__(self, network_params=None, analysis_params=None,
                 new_network_params={}, new_analysis_params={},
                 derive_params=True, file=None):
        """
        Initiate Network class.

        Load parameters from given yaml files using input output handling
        implemented in io.py and store them as instance variables.
        Overwrite parameters specified in new_network_parms and
        new_analysis_params.
        Calculate parameters which are derived from given parameters.
        #Try to load existing results.
        """
        if file:
            self.load(file)
        else:
            # no yaml file for network parameters given
            if network_params is None:
                self.network_params_yaml = ''
                self.network_params = new_network_params
            else:
                self.network_params_yaml = network_params
                # read from yaml and convert to quantities
                self.network_params = io.load_params(network_params)
                self.network_params.update(new_network_params)

            # no yaml file for analysis parameters given
            if analysis_params is None:
                self.analysis_params_yaml = ''
                self.analysis_params = new_analysis_params
            else:
                self.analysis_params_yaml = analysis_params
                self.analysis_params = io.load_params(analysis_params)
                self.analysis_params.update(new_analysis_params)

            if derive_params:
                # calculate dependend network parameters
                derived_network_params = (
                    self._calculate_dependent_network_parameters())
                self.network_params.update(derived_network_params)

                # calculate dependend analysis parameters
                derived_analysis_params = (
                    self._calculate_dependent_analysis_parameters())
                self.analysis_params.update(derived_analysis_params)

            # calc hash
            self.hash = io.create_hash(self.network_params,
                                       self.network_params.keys())
            
            # empty results
            self.results = {}
            self.results_hash_dict = {}

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
                               * self.network_params['w']).to(ureg.mV)

        # weight matrix in pA (current)
        W = np.ones((dim, dim)) * self.network_params['w']
        W[1:dim:2] *= -self.network_params['g']
        W = np.transpose(W)
        derived_params['W'] = W

        # weight matrix in mV (voltage)
        derived_params['J'] = (tau_s_div_C * derived_params['W']).to(ureg.mV)

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

        # TODO: Put calculation of network-specifc dervied parameters
        # into an external script for enhanced generalization.
        # (e.g. trigger execution of external script called <label>.py here)
        # Changing the label currently leads to difference which are hard to
        # track down.
        if self.network_params['label'] == 'microcircuit':
            # larger weight for L4E->L23E connections
            derived_params['W'][0][2] *= 2.0
            derived_params['J'][0][2] *= 2.0

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

        # enable usage of quantities
        @ureg.wraps(ureg.Hz, (ureg.Hz, ureg.Hz, ureg.Hz))
        def calc_evaluated_omegas(w_min, w_max, dw):
            """ Calculates omegas at which functions are to be evaluated """
            return np.arange(w_min, w_max, dw)

        derived_params['omegas'] = calc_evaluated_omegas(w_min, w_max, dw)

        @ureg.wraps((1 / ureg.mm).units,
                    ((1 / ureg.mm).units, (1 / ureg.mm).units,
                     (1 / ureg.mm).units))
        def calc_evaluated_wavenumbers(k_min, k_max, dk):
            return np.arange(k_min, k_max, dk)

        derived_params['k_wavenumbers'] = (
            calc_evaluated_wavenumbers(self.analysis_params['k_min'],
                                       self.analysis_params['k_max'],
                                       self.analysis_params['dk']))

        return derived_params

    def _check_and_store(result_key, analysis_keys=None):
        """
        Decorator function that checks whether result are already existing.

        This decorator serves as a wrapper for functions that calculate
        quantities which are to be stored in self.results. First it checks,
        whether the result already has been stored in self.results. If this is
        the case, it returns that result. If not, the calculation is executed,
        the result is stored in self.results and the result is returned.
        Additionally results are stored in self.results_hash_dict to simplify
        searching.

        If the wrapped function gets additional parameters passed, one should
        also include an analysis key, under which the new analysis parameters
        should be stored in the dictionary self.analysis_params. Then, the
        decorator first checks, whether the given parameters have been used
        before and returns the corresponding results.
        
        This function can only handle unitless objects or quantities. Lists or
        arrays of quantites are not allowed. Use quantity arrays instead (a
        quantity with array magnitude and a unit).

        TODO: Implement possibility to pass list of result_keys

        Parameters:
        -----------
        result_key: str
            Specifies under which key the result should be stored.
        analysis_key: list
            Specifies under which key the analysis_parameters should be stored.

        Returns:
        --------
        func
            decorator function
        """

        @decorator
        def decorator_check_and_store(func, self, *args, **kwargs):
            """ Decorator with given parameters, returns expected results. """
            # collect analysis_params
            analysis_params = getattr(self, 'analysis_params')

            # collect results
            results = getattr(self, 'results')
            results_hash_dict = getattr(self, 'results_hash_dict')

            # convert new params to list
            new_params = []
            if analysis_keys is not None:
                for i, key in enumerate(analysis_keys):
                    new_params.append(args[i])

            # calculate hash from result and analysis keys and analysis params
            label = str(result_key) + str(analysis_keys) + str(new_params)
            h = hashlib.md5(label.encode('utf-8')).hexdigest()
            # h = hash(str(result_key) + str(analysis_keys) + str(new_params))
            # check if hash exists and return existing result if true
            if h in results_hash_dict.keys():
                return results_hash_dict[h]['result']
            else:
                # if not, calculate new result
                result = func(self, *args, **kwargs)

                # store keys and results and update dictionaries
                results[result_key] = result
                hash_dict = dict(result=result,
                                 result_key=result_key)
                if analysis_keys:
                    analysis_dict = {}
                    for key, param in zip(analysis_keys, new_params):
                        analysis_params[key] = param
                        analysis_dict[key] = param
                    hash_dict['analysis_params'] = analysis_dict
                results_hash_dict[h] = hash_dict
                
                # update self.results and self.results_hash_dict
                setattr(self, 'results', results)
                setattr(self, 'results_hash_dict', results_hash_dict)
                setattr(self, 'analysis_params', analysis_params)

                # return new_result
                return result

        return decorator_check_and_store
    
    def save(self, file_name, overwrite_dataset=False):
        """
        Save network to h5 file.
        
        The networks' dictionaires (network_params, analysis_params, results,
        results_hash_dict) are stored. Quantities are converted to value-unit
        dictionaries.
        
        Parameters:
        -----------
        file_name: str
            Output file name.
        overwrite_dataset: bool
            Whether to overwrite an existing h5 file or not. If there already
            is one, h5py tries to update the h5 dictionary.
        """
        io.save_network(file_name, self, overwrite_dataset)
    
    def load(self, file_name):
        """
        Load network from h5 file.
        
        The networks' dictionaires (network_params, analysis_params, results,
        results_hash_dict) are loaded.
        
        Parameters:
        -----------
        file_name: str
            Input file name.
        """
        (self.network_params,
         self.analysis_params,
         self.results,
         self.results_hash_dict) = io.load_network(file_name)
    
    def save_results(self, output_key='', output={}, file_name=''):
        """
        Saves results and parameters to h5 file. If output is specified, this
        is saved to h5 file.

        Parameters:
        -----------
        output_key: str
            if specified, save output_dict under key output name in h5 file
        output: dict
            data that is stored in h5 file
        file_name: str
            if given, this is used as output file name

        Returns:
        --------
        None
        """
        # if no file name is specified use standard version
        if not file_name:
            file_name = '{}_{}.h5'.format(self.network_params['label'],
                                          str(self.hash))

        # if output is given, save it to h5 file
        if output_key:
            io.save(output_key, output, file_name)

        # else save results and parameters to h5 file
        else:
            io.save('results', self.results, file_name)
            io.save('network_params', self.network_params, file_name)
            io.save('analysis_params', self.analysis_params, file_name)

    def show(self):
        """ Returns which results have already been calculated """
        return sorted(list(self.results.keys()))

    def change_parameters(self, changed_network_params={},
                          changed_analysis_params={}):
        """
        Change parameters and return new network with specified parameters.

        Parameters:
        -----------
        new_network_parameters: dict
            Dictionary specifying which parameters should be altered.

        Returns:
        Network object
            New network with specified parameters.
        """

        new_network_params = self.network_params
        new_network_params.update(changed_network_params)
        new_analysis_params = self.analysis_params
        new_analysis_params.update(changed_analysis_params)

        return Network(self.network_params_yaml, self.analysis_params_yaml,
                       new_network_params, new_analysis_params)

    def extend_analysis_frequencies(self, f_min, f_max):
        """
        Extend analysis frequencies and calculate all results for new ranges.

        Paramters:
        ----------
        f_min: Quantity(float, 'Hz')
            Minimal frequency analysed.
        f_max: Quantity(float, 'Hz')
            Maximal frequency analysed.
        """
        pass

    @_check_and_store('firing_rates')
    def firing_rates(self):
        """ Calculates firing rates """
        return meanfield_calcs.firing_rates(self.network_params['dimension'],
                                            self.network_params['tau_m'],
                                            self.network_params['tau_s'],
                                            self.network_params['tau_r'],
                                            self.network_params['V_0_rel'],
                                            self.network_params['V_th_rel'],
                                            self.network_params['K'],
                                            self.network_params['J'],
                                            self.network_params['j'],
                                            self.network_params['nu_ext'],
                                            self.network_params['K_ext'],
                                            self.network_params['g'],
                                            self.network_params['nu_e_ext'],
                                            self.network_params['nu_i_ext'])

    @_check_and_store('mean_input')
    def mean_input(self):
        """ Calculates mean """
        return meanfield_calcs.mean(self.firing_rates(),
                                    self.network_params['K'],
                                    self.network_params['J'],
                                    self.network_params['j'],
                                    self.network_params['tau_m'],
                                    self.network_params['nu_ext'],
                                    self.network_params['K_ext'],
                                    self.network_params['g'],
                                    self.network_params['nu_e_ext'],
                                    self.network_params['nu_i_ext'])

    @_check_and_store('std_input')
    def std_input(self):
        """ Calculates variance """
        return meanfield_calcs.standard_deviation(
            self.firing_rates(),
            self.network_params['K'],
            self.network_params['J'],
            self.network_params['j'],
            self.network_params['tau_m'],
            self.network_params['nu_ext'],
            self.network_params['K_ext'],
            self.network_params['g'],
            self.network_params['nu_e_ext'],
            self.network_params['nu_i_ext'])

    def working_point(self):
        """
        Calculates stationary working point of the network.

        Returns:
        --------
        dict
            dictionary specifying mean, variance and firing rates
        """

        # first define functions that keep track of already existing results

        # then do calculations
        working_point = {}
        working_point['firing_rates'] = self.firing_rates()
        working_point['mean_input'] = self.mean_input()
        working_point['std_input'] = self.std_input()

        return working_point

    def delay_dist_matrix(self, freq=None):
        """
        Calculates delay dist matrix either for all frequencies or given one.

        Paramters:
        ----------
        freq: Quantity(float, 'Hertz')
            Optional paramter. If given, delay dist matrix is only calculated
            for this frequency.

        Returns:
        --------
        Quantity(np.ndarray, 'Hz/mV')
            Delay dist matrix, either as an array with shape(dimension,
            dimension) for a given frequency, or shape(dimension, dimension,
            len(omegas)) for no specified frequency.
        """
        if freq is None:
            return self.delay_dist_matrix_multi()
        else:
            omega = 2 * np.pi * freq
            return self.delay_dist_matrix_single(omega)

    @_check_and_store('delay_dist')
    def delay_dist_matrix_multi(self):
        """
        Calculates delay distribution matrix for all omegas.

        Returns:
        --------
        Quantity(np.ndarray, 'dimensionless'):
            Delay distribution matrix.
        """

        return meanfield_calcs.delay_dist_matrix(
            self.network_params['dimension'],
            self.network_params['Delay'],
            self.network_params['Delay_sd'],
            self.network_params['delay_dist'],
            self.analysis_params['omegas'])

    @_check_and_store('delay_dist_single', ['delay_dist_freqs'])
    def delay_dist_matrix_single(self, omega):
        """
        Calculates delay distribution matrix for one omega.

        Parameters:
        -----------
        omega: Quantity(float, 'Hertz')
            Frequency for which delay distribution matrix should be calculated.
        Returns:
        --------
        Quantity(np.ndarray, 'dimensionless'):
            Delay distribution matrix.
        """
        return meanfield_calcs.delay_dist_matrix(
            self.network_params['dimension'],
            self.network_params['Delay'],
            self.network_params['Delay_sd'],
            self.network_params['delay_dist'],
            [omega])[0]

    def transfer_function(self, freq=None, method='shift'):
        """
        Calculates transfer function either for all frequencies or given one.

        Paramters:
        ----------
        freq: Quantity(float, 'Hertz')
            Optional paramter. If given, transfer function is only calculated
            for this frequency.

        Returns:
        --------
        Quantity(np.ndarray, 'Hz/mV')
            Transfer function, either as an array with shape(dimension,) for a
            given frequency, or shape(dimension, len(omegas)) for no specified
            frequency.
        """
        if freq is None:
            return self.transfer_function_multi(method)
        else:
            return self.transfer_function_single(freq, method)

    @_check_and_store('transfer_function', ['transfer_multi_method'])
    def transfer_function_multi(self, method='shift'):
        """
        Calculates transfer function for each population.

        Returns:
        --------
        Quantity(np.ndarray, 'dimensionless'):
            Transfer functions for all populations evaluated at specified
            omegas.
        """
        transfer_functions = meanfield_calcs.transfer_function(
            self.mean_input(),
            self.std_input(),
            self.network_params['tau_m'],
            self.network_params['tau_s'],
            self.network_params['tau_r'],
            self.network_params['V_th_rel'],
            self.network_params['V_0_rel'],
            self.network_params['dimension'],
            self.analysis_params['omegas'],
            method=method)

        return transfer_functions

    @_check_and_store('transfer_function_single', ['transfer_freqs',
                                                   'transfer_single_method'])
    def transfer_function_single(self, freq, method='shift'):
        """
        Calculates transfer function for each population.

        Returns:
        --------
        Quantity(np.ndarray, 'dimensionless'):
            Transfer functions for all populations evaluated at specified
            omegas.
        """
        omega = freq * 2 * np.pi

        transfer_functions = meanfield_calcs.transfer_function(
            self.mean_input(),
            self.std_input(),
            self.network_params['tau_m'],
            self.network_params['tau_s'],
            self.network_params['tau_r'],
            self.network_params['V_th_rel'],
            self.network_params['V_0_rel'],
            self.network_params['dimension'],
            [omega],
            method=method)

        return transfer_functions

    @_check_and_store('sensitivity_measure', ['sensitivity_freqs'])
    def sensitivity_measure(self, freq, method='shift'):
        """
        Calculates the sensitivity measure for the given frequency.

        Following Eq. 21 in Bos et al. (2015).

        Parameters:
        -----------
        freq: Quantity(float, 'hertz')
            Regular frequency at which sensitivity measure is evaluated.

        Returns:
        --------
        Quantity(np.ndarray, 'dimensionless')
            Sensitivity measure.
        """
        # convert regular frequency to angular frequeny
        omega = freq * 2 * np.pi

        # calculate needed transfer_function
        transfer_function = meanfield_calcs.transfer_function(
            self.mean_input(),
            self.std_input(),
            self.network_params['tau_m'],
            self.network_params['tau_s'],
            self.network_params['tau_r'],
            self.network_params['V_th_rel'],
            self.network_params['V_0_rel'],
            self.network_params['dimension'],
            [omega],
            method=method)

        if omega.magnitude < 0:
            transfer_function = np.conjugate(transfer_function)

        # calculate needed delay distribution matrix
        delay_dist_matrix = meanfield_calcs.delay_dist_matrix(
            self.network_params['dimension'],
            self.network_params['Delay'],
            self.network_params['Delay_sd'],
            self.network_params['delay_dist'],
            [omega])[0]

        return meanfield_calcs.sensitivity_measure(
            transfer_function,
            delay_dist_matrix,
            self.network_params['J'],
            self.network_params['K'],
            self.network_params['tau_m'],
            self.network_params['tau_s'],
            self.network_params['dimension'],
            omega)

    @_check_and_store('power_spectra')
    def power_spectra(self, method='shift'):
        """
        Calculates power spectra.
        """
        return meanfield_calcs.power_spectra(
            self.network_params['tau_m'],
            self.network_params['tau_s'],
            self.network_params['dimension'],
            self.network_params['J'],
            self.network_params['K'],
            self.delay_dist_matrix(),
            self.network_params['N'],
            self.firing_rates(),
            self.transfer_function(method=method),
            self.analysis_params['omegas'])

    @_check_and_store('eigenvalue_spectra', ['eigenvalue_matrix'])
    def eigenvalue_spectra(self, matrix, method='shift'):
        """
        Calculates the eigenvalues of the specified matrix at given frequency.

        Paramters:
        ----------
        matrix: str
            Specifying matrix which is analysed. Options are the effective
            connectivity matrix ('MH'), the propagator ('prop') and
            the inverse of the propagator ('prop_inv').

        Returns:
        --------
        Quantity(np.ndarray, 'dimensionless')
            Eigenvalues.
        """
        return meanfield_calcs.eigen_spectra(
            self.network_params['tau_m'],
            self.network_params['tau_s'],
            self.transfer_function(method=method),
            self.network_params['dimension'],
            self.delay_dist_matrix(),
            self.network_params['J'],
            self.network_params['K'],
            self.analysis_params['omegas'],
            'eigvals',
            matrix)

    @_check_and_store('r_eigenvec_spectra', ['r_eigenvec_matrix'])
    def r_eigenvec_spectra(self, matrix):
        """
        Calculates the right eigenvecs of the specified matrix at given freq.

        Paramters:
        ----------
        matrix: str
            Specifying matrix which is analysed. Options are the effective
            connectivity matrix ('MH'), the propagator ('prop') and
            the inverse of the propagator ('prop_inv').

        Returns:
        --------
        Quantity(np.ndarray, 'dimensionless')
            Right eigenvectors.
        """
        return meanfield_calcs.eigen_spectra(
            self.network_params['tau_m'],
            self.network_params['tau_s'],
            self.transfer_function(),
            self.network_params['dimension'],
            self.delay_dist_matrix(),
            self.network_params['J'],
            self.network_params['K'],
            self.analysis_params['omegas'],
            'reigvecs',
            matrix)

    @_check_and_store('l_eigenvec_spectra', ['l_eigenvec_matrix'])
    def l_eigenvec_spectra(self, matrix):
        """
        Calculates the left eigenvecs of the specified matrix at given freq.

        Paramters:
        ----------
        matrix: str
            Specifying matrix which is analysed. Options are the effective
            connectivity matrix ('MH'), the propagator ('prop') and
            the inverse of the propagator ('prop_inv').

        Returns:
        --------
        Quantity(np.ndarray, 'dimensionless')
            Left eigenvectors.
        """
        return meanfield_calcs.eigen_spectra(
            self.network_params['tau_m'],
            self.network_params['tau_s'],
            self.transfer_function(),
            self.network_params['dimension'],
            self.delay_dist_matrix(),
            self.network_params['J'],
            self.network_params['K'],
            self.analysis_params['omegas'],
            'leigvecs',
            matrix)

    def additional_rates_for_fixed_input(self, mean_input_set, std_input_set):
        """
        Calculate additional external excitatory and inhibitory Poisson input
        rates such that the input prescribed by the mean and standard deviation
        is attained.

        Parameters:
        -----------
        mean_input_set: Quantity(np.ndarray, 'mV')
            prescribed mean input for each population
        std_input_set: Quantity(np.ndarray, 'mV')
            prescribed standard deviation of input for each population

        Returns:
        --------
        nu_e_ext: Quantity(np.ndarray, 'hertz')
            additional external excitatory rate needed for fixed input
        nu_i_i: Quantity(np.ndarray, 'hertz')
            additional external inhibitory rate needed for fixed input
        """
        nu_e_ext, nu_i_ext = \
            meanfield_calcs.additional_rates_for_fixed_input(
                mean_input_set, std_input_set,
                self.network_params['tau_m'],
                self.network_params['tau_s'],
                self.network_params['tau_r'],
                self.network_params['V_0_rel'],
                self.network_params['V_th_rel'],
                self.network_params['K'],
                self.network_params['J'],
                self.network_params['j'],
                self.network_params['nu_ext'],
                self.network_params['K_ext'],
                self.network_params['g'])
        return nu_e_ext, nu_i_ext

    def fit_transfer_function(self):
        """
        Fit the absolute value of the LIF transfer function to the one of a
        first-order low-pass filter. Compute the time constants and weight
        matrices for an equivalent rate model. Compute the effective coupling
        strength for comparison.

        Returns:
        --------
        tau_rate: Quantity(np.ndarray, 'second')
            Time constants for rate model obtained from fit.
        W_rate: np.ndarray
            Weights for rate model obtained from fit.
        W_rate_sim: np.ndarray
            Weights for rate model obtained from fit divided by indegrees,
            to be used in simulation.
        fit_tf: Quantity(np.ndarray, 'hertz/millivolt')
            Fitted transfer function.
        tf0_ecs: Quantity(np.ndarray, 'hertz/millivolt')
            Effective coupling strength scaled to transfer function.
        """
        tau_rate, W_rate, W_rate_sim, fit_tf = (
            meanfield_calcs.fit_transfer_function(
                self.transfer_function(),
                self.analysis_params['omegas'],
                self.network_params['tau_m'],
                self.network_params['J'],
                self.network_params['K']))

        w_ecs = meanfield_calcs.effective_coupling_strength(
            self.network_params['tau_m'],
            self.network_params['tau_s'],
            self.network_params['tau_r'],
            self.network_params['V_0_rel'],
            self.network_params['V_th_rel'],
            self.network_params['J'],
            self.mean_input(),
            self.std_input())

        # scale to transfer function and adapt unit
        tf0_ecs = (w_ecs / (self.network_params['J']
                   * self.network_params['tau_m'])).to(ureg.Hz / ureg.mV)

        return tau_rate, W_rate, W_rate_sim, fit_tf, tf0_ecs

    def scan_fit_transfer_function_mean_std_input(self, mean_inputs,
                                                  std_inputs):
        """
        Scan all combinations of mean_inputs and std_inputs: Compute and fit
        the transfer function for each case and return the relative fit errors
        on tau and h0.

        Parameters:
        -----------
        mean_inputs: Quantity(np.ndarray, 'mV')
            List of mean inputs to scan.
        std_inputs: Quantity(np.ndarray, 'mV')
            List of standard deviation of inputs to scan.

        Returns:
        --------
        errs_tau: np.ndarray
            Relative error on fitted tau for each combination of mean and std
            of input.
        errs_h0: np.ndarray
            Relative error on fitted h0 for each combination of mean and std
            of input.
        """
        errs_tau, errs_h0 = (
            meanfield_calcs.scan_fit_transfer_function_mean_std_input(
                mean_inputs, std_inputs,
                self.network_params['tau_m'],
                self.network_params['tau_s'],
                self.network_params['tau_r'],
                self.network_params['V_0_rel'],
                self.network_params['V_th_rel'],
                self.analysis_params['omegas']))

        return errs_tau, errs_h0

    def linear_interpolation_alpha(self, k_wavenumbers, network):
        """
        Linear interpolation between analytically solved characteristic
        equation for linear rate model and equation solved for lif model.
        Eigenvalues lambda are computed by solving the characteristic equation
        numerically or by solving an integral.
        Reguires a spatially organized network with boxcar connectivity
        profile.

        Parameters:
        -----------
        k_wavenumbers: Quantity(np.ndarray, '1/m')
            Range of wave numbers.
        network: Network object
            A network.

        Returns:
        --------
        alphas: np.ndarray
        lambdas_chareq: Quantity(np.ndarray, '1/s')
        lambdas_integral: Quantity(np.ndarray, '1/s')
        k_eig_max: Quantity(float, '1/m')
        eigenval_max: Quantity(complex, '1/s')
        eigenvals: Quantity(np.ndarray, '1/s')
        """
        (alphas, lambdas_chareq, lambdas_integral, k_eig_max, eigenval_max,
         eigenvals) = (
            meanfield_calcs.linear_interpolation_alpha(
                k_wavenumbers,
                self.analysis_params['branches'],
                self.network_params['tau_rate'],
                self.network_params['W_rate'],
                self.network_params['width'],
                self.network_params['d_e'],
                self.network_params['d_i'],
                self.mean_input(),
                self.std_input(),
                self.network_params['tau_m'],
                self.network_params['tau_s'],
                self.network_params['tau_r'],
                self.network_params['V_0_rel'],
                self.network_params['V_th_rel'],
                self.network_params['J'],
                self.network_params['K'],
                self.network_params['dimension'],
                ))

        return (alphas, lambdas_chareq, lambdas_integral, k_eig_max,
                eigenval_max, eigenvals)

    def compute_profile_characteristics(self):
        """
        """
        xi_min, xi_max, k_min, k_max = (
            meanfield_calcs.xi_of_k(self.analysis_params['k_wavenumbers'],
                                    self.network_params['W_rate'],
                                    self.network_params['width']))

        lambda_min = meanfield_calcs.solve_chareq_rate_boxcar(
            0,  # branch
            k_min,
            self.network_params['tau_rate'][0],
            self.network_params['W_rate'],
            self.network_params['width'],
            self.network_params['d_e']
            )
        lambda_max = meanfield_calcs.solve_chareq_rate_boxcar(
            0,  # branch
            k_max,
            self.network_params['tau_rate'][0],
            self.network_params['W_rate'],
            self.network_params['width'],
            self.network_params['d_e']
            )

        self.results.update({
            'rho': (self.network_params['width'][1].to(ureg.mm).magnitude
                    / self.network_params['width'][0].to(ureg.mm).magnitude),
            'eta': (-1 * self.network_params['W_rate'][0, 1]
                    / self.network_params['W_rate'][0, 0]),
            'xi_min': xi_min,
            'xi_max': xi_max,
            'k_min': k_min,
            'k_f_min': k_min / (2. * np.pi),
            'k_max': k_max,
            'k_f_max': k_max / (2. * np.pi),
            'tau_delay':
                (self.network_params['tau_rate'][0].to(ureg.ms).magnitude
                 / self.network_params['d_e'].to(ureg.ms).magnitude),
            'lambda_min': lambda_min,
            'lambda_max': lambda_max,
            'lambda_f_min': np.imag(lambda_min) / (2. * np.pi),
            'speed': np.imag(lambda_min.to(1 / ureg.s)) / k_min.to(1 / ureg.m),
            })
        return
