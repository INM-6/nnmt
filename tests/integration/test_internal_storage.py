import pytest

import lif_meanfield_tools as lmt


def test_rates_not_calculated_twice_with_same_parameters(network, mocker):
    mocked = mocker.patch(
        'lif_meanfield_tools.lif.exp._firing_rates',
        spec=True,
        return_value=1
        )
    lmt.lif.exp.firing_rates(network)
    lmt.lif.exp.firing_rates(network)
    mocked.assert_called_once()
    
    
def test_rates_calculated_again_after_parameters_are_changed(network, mocker):
    mocked = mocker.patch(
        'lif_meanfield_tools.lif.exp._firing_rates',
        spec=True,
        return_value=1
        )
    lmt.lif.exp.firing_rates(network)
    network.network_params['tau_m'] *= 2
    lmt.lif.exp.firing_rates(network)
    assert mocked.call_count == 2


def test_rates_calculated_again_for_differing_methods(network, mocker):
    mocked_shift = mocker.patch(
        'lif_meanfield_tools.lif.exp._firing_rate_shift',
        spec=True,
        return_value=1
        )
    mocked_taylor = mocker.patch(
        'lif_meanfield_tools.lif.exp._firing_rate_taylor',
        spec=True,
        return_value=2
        )
    lmt.lif.exp.firing_rates(network, method='shift')
    lmt.lif.exp.firing_rates(network, method='taylor')
    assert mocked_shift.called
    assert mocked_taylor.called


def test_units_and_magnitudes_same_before_and_after_saving_and_loading(
        empty_network, mocker, tmpdir):
    
    network = empty_network
    mock = mocker.Mock(__name__='func', return_value=1)
    
    def test_function(network):
        return lmt.utils._cache(network, mock, dict(a=1), 'test', 'millivolt')

    test_function(network)
    value_before = network.results['test']
    unit_before = network.result_units['test']
    
    temp = tmpdir.mkdir('temp')
    with temp.as_cwd():
        # first try
        value_before = network.results['test']
        unit_before = network.result_units['test']
        network.save(file='test.h5', overwrite=True)
        network.load(file='test.h5')
        value_after = network.results['test']
        unit_after = network.result_units['test']
        assert value_before == value_after
        assert unit_before == unit_after
        
        # second try
        value_before = value_after
        unit_before = unit_after
        network.save(file='test.h5', overwrite=True)
        network.load(file='test.h5')
        value_after = network.results['test']
        unit_after = network.result_units['test']
        assert value_before == value_after
        assert unit_before == unit_after
