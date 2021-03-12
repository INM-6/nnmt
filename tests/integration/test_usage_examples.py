import pytest
import h5py_wrapper.wrapper as h5

import lif_meanfield_tools as lmt
from numpy.testing import assert_array_equal
from .test_functionality import assert_quantity_array_equal

config_path = 'tests/fixtures/integration/config/'


@pytest.fixture(scope='class')
def network():
    """Standard microcircuit network with testing analysis params."""
    network = lmt.Network(
        network_params=(config_path + 'network_params.yaml'),
        analysis_params=(config_path + 'analysis_params.yaml')
        )
    return network


class Test_Network_instantiation_calculation_saving_routine:
    
    def test_calculate_something(self, network, std_results):
        firing_rates = network.firing_rates()
        assert_quantity_array_equal(firing_rates, std_results['firing_rates'])
    
    def test_list_results(self, network):
        list_of_results = network.show()
        assert list_of_results == ['firing_rates']
    
    def test_look_at_results(self, network, std_results):
        assert_quantity_array_equal(network.results['firing_rates'],
                                    std_results['firing_rates'])
        
    def test_save_something(self, network, tmpdir, std_results):
        temp = tmpdir.mkdir('temp')
        with temp.as_cwd():
            network.save(file_name='test.h5')
            loaded = h5.load('test.h5')
        firing_rates = loaded['results']['firing_rates']['val']
        assert_array_equal(firing_rates, std_results['firing_rates'])
        

class Test_instantiate_with_some_passed_quantities_calculate_save_routine:

    def test_instantiate_network_with_passed_params(self, network,
                                                    network_params_yaml,
                                                    analysis_params_yaml):
        network_params = network.network_params
        tau_m = network_params['tau_m']
        analysis_params = network.analysis_params
        omega = analysis_params['omega']
        network = lmt.Network(network_params_yaml, analysis_params_yaml,
                              new_network_params=dict(tau_m=2 * tau_m),
                              new_analysis_params=dict(omega=2 * omega))
        assert network.network_params['tau_m'] == 2 * tau_m
        assert network.analysis_params['omega'] == 2 * omega
        
    def test_calculate_something(self, network, std_results):
        firing_rates = network.firing_rates()
        assert_quantity_array_equal(firing_rates, std_results['firing_rates'])

    def test_list_results(self, network):
        list_of_results = network.show()
        assert list_of_results == ['firing_rates']
        
    def test_look_at_results(self, network, std_results):
        assert_quantity_array_equal(network.results['firing_rates'],
                                    std_results['firing_rates'])
        
    def test_save_something(self, network, tmpdir, std_results):
        temp = tmpdir.mkdir('temp')
        with temp.as_cwd():
            network.save(file_name='test.h5')
            loaded = h5.load('test.h5')
        firing_rates = loaded['results']['firing_rates']['val']
        assert_array_equal(firing_rates, std_results['firing_rates'])
        
        
class Test_instantiate_with_passed_dictionaries_calculate_save_routine:

    def test_instantiate_network_with_passed_params(self, network,
                                                    network_params_yaml,
                                                    analysis_params_yaml):
        network_params = network.network_params
        tau_m = network_params['tau_m']
        network_params['tau_m'] *= 2
        analysis_params = network.analysis_params
        omega = analysis_params['omega']
        analysis_params['omega'] *= 2
        network = lmt.Network(new_network_params=network_params,
                              new_analysis_params=analysis_params)
        assert network.network_params['tau_m'] == 2 * tau_m
        assert network.analysis_params['omega'] == 2 * omega


class Test_instantiate_calculate_check_change_params_calculate_check:

    def test_calculate_something(self, network, std_results):
        firing_rates = network.firing_rates()
        assert_quantity_array_equal(firing_rates, std_results['firing_rates'])

    def test_list_results(self, network):
        list_of_results = network.show()
        assert list_of_results == ['firing_rates']

    def test_look_at_results(self, network, std_results):
        assert_quantity_array_equal(network.results['firing_rates'],
                                    std_results['firing_rates'])
        
    def test_change_parameters(self, network):
        tau_m = network.network_params['tau_m']
        network.change_parameters(changed_network_params=dict(tau_m=2 * tau_m))
        assert network.network_params['tau_m'] == 2 * tau_m
        
    def test_calculate_something_again_get_different_results(self,
                                                             network,
                                                             std_results):
        firing_rates = network.firing_rates()
        with pytest.raises(AssertionError):
            assert_quantity_array_equal(firing_rates,
                                        std_results['firing_rates'])

    def test_look_at_results_again(self, network, std_results):
        with pytest.raises(AssertionError):
            assert_quantity_array_equal(network.results['firing_rates'],
                                        std_results['firing_rates'])

    def test_save_something(self, network, tmpdir, std_results):
        temp = tmpdir.mkdir('temp')
        with temp.as_cwd():
            network.save(file_name='test.h5')
            loaded = h5.load('test.h5')
        firing_rates = loaded['results']['firing_rates']['val']
        with pytest.raises(AssertionError):
            assert_array_equal(firing_rates, std_results['firing_rates'])


class Test_instantiate_calculate_several_things_show_and_check_save:

    def test_calculate_something(self, network, std_results):
        firing_rates = network.firing_rates()
        assert_quantity_array_equal(firing_rates, std_results['firing_rates'])
        
    def test_calculate_something_else(self, network, std_results):
        mean_input = network.mean_input()
        assert_quantity_array_equal(mean_input, std_results['mean_input'])

    def test_list_what_has_been_calculated(self, network):
        list_of_results = network.show()
        assert list_of_results == ['firing_rates', 'mean_input']

    def test_look_at_results(self, network, std_results):
        assert_quantity_array_equal(network.results['firing_rates'],
                                    std_results['firing_rates'])
        assert_quantity_array_equal(network.results['mean_input'],
                                    std_results['mean_input'])
