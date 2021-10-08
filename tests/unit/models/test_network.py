import pytest
import numpy as np
import nnmt.input_output as io

from ...checks import (assert_array_equal,
                       assert_quantity_array_equal,
                       check_quantity_dicts_are_equal)

import nnmt

ureg = nnmt.ureg


class Test_initialization:

    def test_all_dicts_created(self):
        network = nnmt.models.Network()
        assert hasattr(network, 'network_params')
        assert hasattr(network, 'analysis_params')
        assert hasattr(network, 'results')
        assert hasattr(network, 'results_hash_dict')
        assert hasattr(network, 'input_units')
        assert hasattr(network, 'result_units')

    @pytest.mark.parametrize('key, value', [('tau_m', 0.01),
                                            ('d_i_sd', 0.000375),
                                            ('label', 'microcircuit')])
    def test_correct_network_params_loaded(self, network, key, value):
        assert network.network_params[key] == value

    @pytest.mark.parametrize('key, value', [('f_min', 0.1),
                                            ('omega', 20),
                                            ('k_max', 100500.0)])
    def test_correct_analysis_params_loaded(self, network, key, value):
        assert network.analysis_params[key] == value

    def test_loading_of_existing_results(self, unit_fixture_path):
        network = nnmt.models.Network(
            file=f'{unit_fixture_path}test_network.h5')
        assert len(network.network_params.items()) != 0
        assert len(network.analysis_params.items()) != 0
        assert 'lif.exp.firing_rates' in network.results.keys()


class Test_unit_stripping:

    def test_conversion_to_si_units_and_stripping_units(self, empty_network):
        empty_network.network_params = dict(
            a=10 * ureg.ms,
            b=np.ones(3) * ureg.mV / ureg.ms,
            )
        empty_network.analysis_params = dict(
            omega=10 * ureg.Hz,
            dk=10 / ureg.mm
            )
        empty_network._convert_param_dicts_to_base_units_and_strip_units()
        assert empty_network.network_params['a'] == 0.01
        assert_array_equal(empty_network.network_params['b'], np.ones(3))
        assert empty_network.analysis_params['omega'] == 10
        assert empty_network.analysis_params['dk'] == 10000

    def test_input_units_collected(self, empty_network):
        empty_network.network_params = dict(
            a=10 * ureg.ms,
            b=np.ones(3) * ureg.mV / ureg.ms,
            )
        empty_network.analysis_params = dict(
            omega=10 * ureg.Hz,
            dk=10 / ureg.mm
            )
        empty_network._convert_param_dicts_to_base_units_and_strip_units()
        assert empty_network.input_units['a'] == 'millisecond'
        assert empty_network.input_units['b'] == 'millivolt / millisecond'
        assert empty_network.input_units['omega'] == 'hertz'
        assert empty_network.input_units['dk'] == '1 / millimeter'

    def test_result_units_stripped(self, empty_network):
        empty_network.results['test'] = 10 * ureg.ms
        empty_network._strip_result_units()
        assert empty_network.results['test'] == 10


class Test_adding_units_again:

    def test_units_are_added_correctly(self, empty_network):
        empty_network.network_params['a'] = 0.01
        empty_network.network_params['b'] = np.ones(3)
        empty_network.analysis_params['omega'] = 10
        empty_network.analysis_params['dk'] = 10000
        empty_network.input_units['a'] = 'millisecond'
        empty_network.input_units['b'] = 'millivolt / millisecond'
        empty_network.input_units['omega'] = 'hertz'
        empty_network.input_units['dk'] = '1 / millimeter'
        empty_network._add_units_to_param_dicts_and_convert_to_input_units()
        assert empty_network.network_params['a'] == 10 * ureg.ms
        assert_quantity_array_equal(empty_network.network_params['b'],
                                    np.ones(3) * ureg.mV / ureg.ms)
        assert empty_network.analysis_params['omega'] == 10 * ureg.Hz
        assert empty_network.analysis_params['dk'] == 10 / ureg.mm

    def test_result_units_are_added_correctly(self, mocker, empty_network):
        network = empty_network
        mock = mocker.Mock(__name__='mocker', return_value=1 * ureg.mV)

        def test_function(network):
            return nnmt.utils._cache(network, mock, dict(a=1), 'test')

        test_function(network)
        network._add_result_units()
        expected = dict(test=1 * ureg.mV)
        assert_quantity_array_equal(expected, network.results)


class Test_saving_and_loading:

    def test_save_created_output_file_with_results(self, tmpdir,
                                                   empty_network):
        empty_network.results['test'] = 1
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            empty_network.save('test.h5')
            output = nnmt.input_output.load_val_unit_dict_from_h5('test.h5')
            assert 'test' in output['results'].keys()

    @pytest.mark.xfail
    def test_save_overwriting_existing_file_raises_error(self, tmpdir,
                                                         empty_network):
        empty_network.results['test'] = 1
        file = 'file.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            with pytest.raises(IOError):
                empty_network.save(file=file)
                empty_network.save(file=file)

    @pytest.mark.xfail
    def test_save_overwrites_existing_file_if_explicitely_told(
            self, tmpdir, empty_network):
        file = 'file.h5'
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            empty_network.results['test'] = 1
            empty_network.save(file=file)
            empty_network.results['test'] = 2
            empty_network.save(file=file, overwrite=True)
            output = nnmt.input_output.load_val_unit_dict_from_h5(file)
            assert_array_equal(output['results']['test'], 2)

    def test_load_correctly_sets_network_dictionaries(self, tmpdir,
                                                      empty_network):
        network = empty_network
        network.network_params['test'] = 1
        network.analysis_params['test'] = 2
        network.results['test'] = 3
        network.results_hash_dict['test'] = 4
        network.result_units['test'] = 'millivolt'
        nparams = network.network_params
        aparams = network.analysis_params
        results = network.results
        rhd = network.results_hash_dict
        result_units = network.result_units
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            network.save('test.h5')
            network.network_params = {}
            network.analysis_params = {}
            network.results = {}
            network.results_hash_dict = {}
            network.result_units = {}
            network.load('test.h5')
            check_quantity_dicts_are_equal(nparams, network.network_params)
            check_quantity_dicts_are_equal(aparams, network.analysis_params)
            check_quantity_dicts_are_equal(results, network.results)
            check_quantity_dicts_are_equal(rhd, network.results_hash_dict)
            check_quantity_dicts_are_equal(result_units, network.result_units)

    def test_save_adds_units_to_results(self, mocker, tmpdir, empty_network):
        empty_network.results['test'] = 1
        empty_network.result_units['test'] = 'millivolt'
        empty_network.save
        tmp_test = tmpdir.mkdir('tmp_test')
        with tmp_test.as_cwd():
            empty_network.save('test.h5')
            data = io.load_h5('test.h5')
            assert data['results']['test']['val'] == 1
            assert data['results']['test']['unit'] == 'millivolt'


class Test_meta_functions:

    def test_show(self, empty_network):
        assert empty_network.show() == []

        empty_network.results['spam'] = 1
        empty_network.results['ham'] = 2

        assert empty_network.show() == ['ham', 'spam']

    def test_change_network_parameters(self, network):
        new_tau_m = 1000 * ureg.ms
        update = dict(tau_m=new_tau_m)
        network.change_parameters(changed_network_params=update,
                                  overwrite=True)
        assert network.network_params['tau_m'] == new_tau_m

    def test_change_analysis_parameters(self, network):
        new_df = 1000 * ureg.Hz
        update = dict(df=new_df)
        network.change_parameters(changed_analysis_params=update,
                                  overwrite=True)
        assert network.analysis_params['df'] == new_df

    def test_change_parameters_returns_new_network(self, network):
        new_tau_m = 1000 * ureg.ms
        update = dict(tau_m=new_tau_m)
        new_network = network.change_parameters(changed_network_params=update)
        assert new_network is not network

    def test_change_parameters_returns_new_network_with_uncoupled_dicts(
            self, network):
        new_tau_m = 1000 * ureg.ms
        update = dict(tau_m=new_tau_m)
        new_network = network.change_parameters(changed_network_params=update)
        new_network.network_params['K'] = np.array([1, 2, 3])
        new_network.analysis_params['omegas'] = np.array([1, 2, 3]) * ureg.Hz
        with pytest.raises(AssertionError):
            assert_array_equal(network.network_params['K'],
                               new_network.network_params['K'])
        with pytest.raises(AssertionError):
            assert_array_equal(network.analysis_params['omegas'],
                               new_network.analysis_params['omegas'])

    def test_change_parameter_deletes_results_if_overwrite_true(self, network):
        new_tau_m = 1000 * ureg.ms
        update = dict(tau_m=new_tau_m)
        network.change_parameters(changed_network_params=update,
                                  overwrite=True)
        assert len(network.results.items()) == 0
        assert len(network.results_hash_dict.items()) == 0


class Test_clear_results:

    def test_clear_all(self, empty_network):
        a = 1
        b = 2
        empty_network.results['a'] = a
        empty_network.results['b'] = b
        empty_network.results_hash_dict['asdjfkl'] = {'a': a,
                                                      'params': []}
        empty_network.results_hash_dict['lkjhfds'] = {'b': b,
                                                      'params': []}
        empty_network.clear_results()
        assert empty_network.results == {}
        assert empty_network.results_hash_dict == {}

    def test_clear_one(self, empty_network):
        a = 1
        b = 2
        empty_network.results['a'] = a
        empty_network.results['b'] = b
        empty_network.results_hash_dict['asdjfkl'] = {'a': a,
                                                      'params': []}
        empty_network.results_hash_dict['lkjhfds'] = {'b': b,
                                                      'params': []}
        empty_network.clear_results('a')
        assert 'b' in empty_network.results.keys()
        assert 'a' not in empty_network.results.keys()
        assert 'lkjhfds' in empty_network.results_hash_dict.keys()
        assert 'asdjfkl' not in empty_network.results_hash_dict.keys()

    def test_clear_two(self, empty_network):
        a = 1
        b = 2
        c = 3
        empty_network.results['a'] = a
        empty_network.results['b'] = b
        empty_network.results['c'] = c
        empty_network.results_hash_dict['asdjfkl'] = {'a': a,
                                                      'params': []}
        empty_network.results_hash_dict['lkjhfds'] = {'b': b,
                                                      'params': []}
        empty_network.results_hash_dict['jopasdf'] = {'c': c,
                                                      'params': []}
        empty_network.clear_results(['a', 'b'])
        assert 'a' not in empty_network.results.keys()
        assert 'b' not in empty_network.results.keys()
        assert 'c' in empty_network.results.keys()
        assert 'asdjfkl' not in empty_network.results_hash_dict.keys()
        assert 'lkjhfds' not in empty_network.results_hash_dict.keys()
        assert 'jopasdf' in empty_network.results_hash_dict.keys()
