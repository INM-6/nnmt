import copy

from .. import ureg
from .. import input_output as io


class Network():
    """
    Basic Network class with given network and analysis parameters.
    
    This class serves as a container for the results calculated using the
    toolbox. It has five dictionaries:
    - `network_params` contains the network parameters.
    - `analysis_params` contains the analysis parameters.
    - `results_hash_dict` contains all calculated results stored using a hash
    which allows a unique retrieval despite using different methods for
    calculating the same quantity.
    - `results` contains the latest calculated results. Note that only the
    latest results for a given quantity are stored. So if you calculate the
    same quantity using different methods, only the last one will be found in
    `results`.
    - `input_units` where the units of input parameters are stored for
    conversions.

    Parameters:
    -----------
    network_params: str
        Specifies path to yaml file containing network parameters.
    analysis_params: str
        Specifies path to yaml file containing analysis parameters.
    file: str
        File name of h5 file from which network can be loaded. Default is
        `None`.
        
    Methods:
    --------
    save:
        Save network to h5 file.
    save_results:
        Saves results and parameters to h5 file.
    load:
        Load network from h5 file.
    show:
        Returns which results have already been calculated.
    change_parameters:
        Change parameters and return network with specified parameters.
    """
    
    def __init__(self, network_params=None, analysis_params=None, file=None):
        if file:
            self.load(file)
        else:
            # no yaml file for network parameters given
            if network_params is None:
                self.network_params_yaml = ''
                self.network_params = {}
            else:
                self.network_params_yaml = network_params
                # read from yaml and convert to quantities
                self.network_params = io.load_val_unit_dict_from_yaml(
                    network_params)

            # no yaml file for analysis parameters given
            if analysis_params is None:
                self.analysis_params_yaml = ''
                self.analysis_params = {}
            else:
                self.analysis_params_yaml = analysis_params
                self.analysis_params = io.load_val_unit_dict_from_yaml(
                    analysis_params)
                        
            # empty results
            self.results = {}
            self.results_hash_dict = {}
            
            # input unit dict
            self.input_units = {}
            
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
        
        Parameters:
        -----------
        dict: dict
            Dictionary to be converted.
            
        Returns:
        --------
        dict
            Converted dictionary.
        """
        dict = copy.deepcopy(dict)
        for key in dict.keys():
            try:
                input_unit = self.input_units[key]
                try:
                    base_unit = ureg.parse_unit_name(input_unit)[0][1]
                except IndexError:
                    base_unit = str(ureg(input_unit).to_base_units().units)
                quantity = ureg.Quantity(dict[key], base_unit)
                quantity.ito(input_unit)
                dict[key] = quantity
            except KeyError:
                pass
        return dict

    def save(self, file, overwrite=False):
        """
        Save network to h5 file.
        
        The networks' dictionaires (network_params, analysis_params, results,
        results_hash_dict) are stored. Quantities are converted to value-unit
        dictionaries.
        
        Parameters:
        -----------
        file: str
            Output file name.
        overwrite: bool
            Whether to overwrite an existing h5 file or not. If there already
            is one, h5py tries to update the h5 dictionary.
        """
        self._add_units_to_param_dicts_and_convert_to_input_units()
        io.save_network(file, self, overwrite)
        self._convert_param_dicts_to_base_units_and_strip_units()
        
    def save_results(self, file):
        """
        Saves results and parameters to h5 file.

        Parameters:
        -----------
        file: str
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
        
        Parameters:
        -----------
        file: str
            Input file name.
        """
        (self.network_params,
         self.analysis_params,
         self.results,
         self.results_hash_dict) = io.load_network(file)
        self._convert_param_dicts_to_base_units_and_strip_units()

    def show(self):
        """Returns which results have already been calculated."""
        return sorted(list(self.results.keys()))

    def change_parameters(self, changed_network_params={},
                          changed_analysis_params={},
                          overwrite=False):
        """
        Change parameters and return network with specified parameters.

        Parameters:
        -----------
        changed_network_params: dict
            Dictionary specifying which network parameters should be altered.
        changed_analysis_params: dict
            Dictionary specifying which analysis parameters should be altered.
        overwrite: bool
            Specifying whether existing network should be overwritten. Note:
            This deletes the existing results!

        Returns:
        Network object
            New network with specified parameters.
        """
        
        new_network_params = self.network_params.copy()
        new_network_params.update(changed_network_params)
        new_analysis_params = self.analysis_params.copy()
        new_analysis_params.update(changed_analysis_params)
        
        if overwrite:
            self.network_params = new_network_params
            self.analysis_params = new_analysis_params
            # delete results, because otherwise get inconsistens return values
            # from _check_and_store. We do not keep track, which quantities
            # have been recalculated or not.
            self.results = {}
            self.results_hash_dict = {}
            return self
        else:
            return Network(self.network_params_yaml, self.analysis_params_yaml,
                           new_network_params, new_analysis_params)

    def clear_results(self, results=None):
        """
        Remove calculated results or specified ones from internal dicts.
        
        Parameters
        ----------
        results: None or list
            List of results to be removed. Default is None.
        """
        if results is not None:
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
