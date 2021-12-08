"""
Module that contains the basic Network class.
"""

import copy
import numpy as np

from .. import ureg
from .. import input_output as io
from .. import utils


class Network():
    """
    Basic Network parent class all other models inherit from.

    This class serves as a container for network parameters, analysis
    parameters, and results calculated using the toolbox. It has convenient
    saving and loading methods.

    Parameters
    ----------
    network_params : [str | dict], optional
        Path to yaml file containing network parameters or dictionary of
        network parameters.
    analysis_params : [str | dict], optional
        Path to yaml file containing analysis parameters or dictionary of
        analysis parameters.
    file : str, optional
        File name of h5 file from which network can be loaded. Default is
        ``None``.

    Attributes
    ----------
    analysis_params : dict
        Collection of parameters needed for analysing the network model. For
        example the frequencies a quantity should be calculated for.
    analysis_params_yaml : str
        File name of yaml analysis parameter file that is read in and converted
        to ``analysis_params``.
    input_units : dict
        Any read in quantities are converted to SI units and the original units
        are stored in this this dictionary.
    network_params : dict
        Collection of network parameters like numbers of neurons, etc.
    network_params_yaml : str
        File name of yaml network parameter file that is read in and coverted
        to ``network_params``.
    results : dict
        This dictionary stores the most recently calculated results. If an
        already calculated quantity is calculated with another method, the
        new version is stored here. Functions needing some previosly calculated
        results search for them in this dictionary.
    results_hash_dict : dict
        This dictionary stores all calcluated results using a unique hash. When
        a quantity that already has been calculated is to be calculated another
        time, the result is retrieved from this dictionary.
    result_units : dict
        This is where the units of the results are stored. They are retrieved
        when saving results.


    Methods
    -------
    save
        Save network to h5 file.
    save_results
        Saves results and parameters to h5 file.
    load
        Load network from h5 file.
    show
        Returns which results have already been calculated.
    change_parameters
        Change parameters and return network with specified parameters.
    copy
        Returns a deep copy of the network.
    """

    def __init__(self, network_params=None, analysis_params=None, file=None):

        self.input_units = {}
        self.result_units = {}

        if file:
            self.load(file)
        else:
            # no yaml file for network parameters given
            if network_params is None:
                self.network_params_yaml = ''
                self.network_params = {}
            elif isinstance(network_params, str):
                self.network_params_yaml = network_params
                # read from yaml and convert to quantities
                self.network_params = io.load_val_unit_dict_from_yaml(
                    network_params)
            elif isinstance(network_params, dict):
                self.network_params = network_params
            else:
                raise ValueError('Invalid value for `network_params`.')

            # no yaml file for analysis parameters given
            if analysis_params is None:
                self.analysis_params_yaml = ''
                self.analysis_params = {}
            elif isinstance(analysis_params, str):
                self.analysis_params_yaml = analysis_params
                self.analysis_params = io.load_val_unit_dict_from_yaml(
                    analysis_params)
            elif isinstance(analysis_params, dict):
                self.analysis_params = analysis_params
            else:
                raise ValueError('Invalid value for `network_params`.')

            # empty results
            self.results = {}
            self.results_hash_dict = {}

    def _convert_param_dicts_to_base_units_and_strip_units(self):
        """
        Converts the parameter dicts to base units and strips the units.
        """
        for dict in [self.network_params, self.analysis_params]:
            for key in dict.keys():
                try:
                    quantity = dict[key]
                    self.input_units[key] = str(quantity.units)
                    dict[key] = quantity.to_base_units().magnitude
                except AttributeError:
                    pass

    def _add_units_to_param_dicts_and_convert_to_input_units(self):
        """
        Adds units to the parameter dicts and converts them to input units.
        """
        self.network_params = (
            self._add_units_to_dict_and_convert_to_input_units(
                self.network_params))
        self.analysis_params = (
            self._add_units_to_dict_and_convert_to_input_units(
                self.analysis_params))

    def _add_units_to_dict_and_convert_to_input_units(self, dict):
        """
        Adds units to a unitless dict and converts them to input units.

        Parameters
        ----------
        dict : dict
            Dictionary to be converted.

        Returns
        -------
        dict
            Converted dictionary.
        """
        dict = copy.deepcopy(dict)
        for key in dict.keys():
            try:
                input_unit = self.input_units[key]
                dict[key] = utils._convert_from_si_to_prefixed(dict[key],
                                                               input_unit)
            except KeyError:
                pass
        return dict

    def _add_result_units(self):
        """
        Adds units stored in networks result_units dict to results dict.
        """

        for key, unit in self.result_units.items():
            self.results[key] = ureg.Quantity(self.results[key], unit)

    def _strip_result_units(self):
        """
        Converts units to SI and strips units from results dict.
        """

        for key, value in self.results.items():
            if isinstance(value, ureg.Quantity):
                self.results[key] = value.magnitude
                self.result_units[key] = str(value.units)

    def save(self, file):
        """
        Save network to h5 file.

        The networks' dictionaires (network_params, analysis_params, results,
        results_hash_dict) are stored. Quantities are converted to value-unit
        dictionaries.

        Parameters
        ----------
        file : str
            Output file name.
        """
        self._add_units_to_param_dicts_and_convert_to_input_units()
        self._add_result_units()
        io.save_network(file, self)
        self._convert_param_dicts_to_base_units_and_strip_units()
        self._strip_result_units()

    def save_results(self, file):
        """
        Saves results and parameters to h5 file.

        Parameters
        ----------
        file : str
            Output file name.
        """
        self._add_units_to_param_dicts_and_convert_to_input_units()
        output = dict(results=self.results,
                      network_params=self.network_params,
                      analysis_params=self.analysis_params)
        io.save_quantity_dict_to_h5(file, output)
        self._convert_param_dicts_to_base_units_and_strip_units()

    def load(self, file):
        """
        Load network from h5 file.

        The networks' dictionaires (network_params, analysis_params, results,
        results_hash_dict) are loaded.

        Note: The network's state is overwritten!

        Parameters
        ----------
        file : str
            Input file name.
        """
        (self.network_params,
         self.analysis_params,
         self.results,
         self.results_hash_dict) = io.load_network(file)
        self._convert_param_dicts_to_base_units_and_strip_units()
        self._strip_result_units()

    def show(self):
        """Returns which results have already been calculated."""
        return sorted(list(self.results.keys()))

    def change_parameters(self, changed_network_params={},
                          changed_analysis_params={},
                          overwrite=False):
        """
        Change parameters and return network with specified parameters.

        Note: Do not change parameters that are calculated on network
        instantiation. This will lead to inconsistencies.

        Parameters
        ----------
        changed_network_params : dict
            Dictionary specifying which network parameters should be altered.
        changed_analysis_params : dict
            Dictionary specifying which analysis parameters should be altered.
        overwrite : bool
            Specifying whether existing network should be overwritten. Note:
            This deletes the existing results!

        Returns
        -------
        Network object
            New network with specified parameters.
        """

        new_network_params = self.network_params.copy()
        new_network_params.update(changed_network_params)
        new_analysis_params = self.analysis_params.copy()
        new_analysis_params.update(changed_analysis_params)
        self._add_units_to_param_dicts_and_convert_to_input_units()

        if overwrite:
            # delete results, because otherwise get inconsistens return values
            # from _check_and_store. We do not keep track, which quantities
            # have been recalculated or not.
            self.results = {}
            self.results_hash_dict = {}
            self.result_units = {}

            self = self.__init__(new_network_params, new_analysis_params)
        else:
            return self._instantiate(new_network_params, new_analysis_params)

    def _instantiate(self, new_network_params, new_analysis_params):
        """
        Helper method for change of parameters that instatiates network.

        Needs to be implemented for each child class seperately.
        """
        return Network(new_network_params, new_analysis_params)

    def copy(self):
        """
        Returns a deep copy of the network.
        """
        network = Network()
        network.network_params = copy.deepcopy(self.network_params)
        network.analysis_params = copy.deepcopy(self.analysis_params)
        network.results = copy.deepcopy(self.results)
        network.results_hash_dict = copy.deepcopy(self.results_hash_dict)
        network.input_units = copy.deepcopy(self.input_units)
        network.result_units = copy.deepcopy(self.result_units)
        return network

    def clear_results(self, results=None):
        """
        Remove calculated results or specified ones from internal dicts.

        Parameters
        ----------
        results : [None | list]
            List of results to be removed. Default is None.
        """
        if results is not None:
            results = np.atleast_1d(results).tolist()
            hashs = []
            for result in results:
                for hash in self.results_hash_dict.keys():
                    if result in self.results_hash_dict[hash]:
                        hashs.append(hash)
                        self.results.pop(result)
            [self.results_hash_dict.pop(hash) for hash in hashs]
        else:
            self.results_hash_dict = {}
            self.results = {}
