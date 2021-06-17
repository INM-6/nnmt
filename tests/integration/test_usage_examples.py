import pytest
import h5py_wrapper.wrapper as h5

import nnmt
from ..checks import assert_array_equal
from numpy.testing import assert_array_equal

config_path = 'tests/fixtures/integration/config/'


@pytest.fixture(scope='class')
def network():
    """
    Standard microcircuit network with testing analysis params.
    
    Because of the pytest scope the same network will be used for all tests of
    a single test class.
    """
    network = nnmt.models.Microcircuit(
        network_params=(config_path + 'network_params.yaml'),
        analysis_params=(config_path + 'analysis_params.yaml')
        )
    # add a place to store results independently
    return network


class Test_Network_instantiation_calculation_saving_routine:
    
    def test_calculate_something_and_list_results(self, network):
        network.temp_results = nnmt.lif.exp.firing_rates(network)
        list_of_results = network.show()
        assert list_of_results == ['lif.exp.firing_rates']
    
    def test_look_at_results(self, network):
        assert_array_equal(network.results['lif.exp.firing_rates'],
                           network.temp_results)
        
    def test_save_something(self, network, tmpdir):
        temp = tmpdir.mkdir('temp')
        with temp.as_cwd():
            network.save(file='test.h5')
            loaded = h5.load('test.h5')
        firing_rates = loaded['results']['lif.exp.firing_rates']['val']
        assert_array_equal(firing_rates, network.temp_results)
        

class Test_instantiate_with_some_passed_quantities_calculate_save_routine:

    def test_instantiate_network_with_passed_params(self, network,
                                                    network_params_yaml,
                                                    analysis_params_yaml):
        network_params = network.network_params
        tau_m = network_params['tau_m']
        network_params['tau_m'] *= 2
        analysis_params = network.analysis_params
        omega = analysis_params['omega']
        analysis_params['omega'] *= 2
        network = nnmt.models.Microcircuit(
            network_params, analysis_params)
        assert network.network_params['tau_m'] == 2 * tau_m
        assert network.analysis_params['omega'] == 2 * omega
        
    def test_calculate_something_and_list_results(self, network):
        network.temp_results = nnmt.lif.exp.firing_rates(network)
        list_of_results = network.show()
        assert list_of_results == ['lif.exp.firing_rates']
        
    def test_look_at_results(self, network, std_results):
        assert_array_equal(network.results['lif.exp.firing_rates'],
                           network.temp_results)
        
    def test_save_something(self, network, tmpdir, std_results):
        temp = tmpdir.mkdir('temp')
        with temp.as_cwd():
            network.save(file='test.h5')
            loaded = h5.load('test.h5')
        firing_rates = loaded['results']['lif.exp.firing_rates']['val']
        assert_array_equal(firing_rates, network.temp_results)
        
        
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
        network = nnmt.models.Microcircuit(
            network_params=network_params,
            analysis_params=analysis_params)
        assert network.network_params['tau_m'] == 2 * tau_m
        assert network.analysis_params['omega'] == 2 * omega


class Test_instantiate_calculate_check_change_params_calculate_check:

    def test_calculate_something_and_list_results(self, network):
        network.temp_results = nnmt.lif.exp.firing_rates(network)
        list_of_results = network.show()
        assert list_of_results == ['lif.exp.firing_rates']

    def test_look_at_results(self, network):
        assert_array_equal(network.results['lif.exp.firing_rates'],
                           network.temp_results)
        
    def test_change_parameters(self, network):
        tau_m = network.network_params['tau_m']
        network.change_parameters(
            changed_network_params=dict(tau_m=2 * tau_m), overwrite=True)
        assert network.network_params['tau_m'] == 2 * tau_m
        
    def test_calculate_something_again_get_different_results(self, network):
        firing_rates = nnmt.lif.exp.firing_rates(network)
        with pytest.raises(AssertionError):
            assert_array_equal(firing_rates, network.temp_results)

    def test_look_at_results_again(self, network):
        with pytest.raises(AssertionError):
            assert_array_equal(network.results['lif.exp.firing_rates'],
                               network.temp_results)

    def test_save_something(self, network, tmpdir):
        temp = tmpdir.mkdir('temp')
        with temp.as_cwd():
            network.save(file='test.h5')
            loaded = h5.load('test.h5')
        firing_rates = loaded['results']['lif.exp.firing_rates']['val']
        with pytest.raises(AssertionError):
            assert_array_equal(firing_rates, network.temp_results)


class Test_instantiate_calculate_several_things_show_and_check_save:

    def test_calculate_two_things_and_list_results(self, network):
        network.temp_rates = nnmt.lif.exp.firing_rates(network)
        network.temp_mean = nnmt.lif.exp.mean_input(network)
        list_of_results = network.show()
        assert list_of_results == ['lif.exp.firing_rates',
                                   'lif.exp.mean_input']

    def test_look_at_results(self, network):
        assert_array_equal(network.results['lif.exp.firing_rates'],
                           network.temp_rates)
        assert_array_equal(network.results['lif.exp.mean_input'],
                           network.temp_mean)
        
        
class Test_instantiate_calculate_save_load:
    
    def test_calculate_something_and_list_results(self, network, std_results):
        network.temp_results = nnmt.lif.exp.firing_rates(network)
        list_of_results = network.show()
        assert list_of_results == ['lif.exp.firing_rates']
    
    def test_look_at_results(self, network):
        assert_array_equal(network.results['lif.exp.firing_rates'],
                           network.temp_results)
        
    def test_save_and_load_something(self, network, tmpdir):
        temp = tmpdir.mkdir('temp')
        with temp.as_cwd():
            network.save(file='test.h5')
            network.load('test.h5')
            firing_rates = network.results['lif.exp.firing_rates']
            assert_array_equal(firing_rates, network.temp_results)
