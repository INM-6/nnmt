import numpy as np
import yaml

import lif_meanfield_tools as lmt
ureg = lmt.ureg

fixture_path = 'tests/fixtures/'

cases = [0, 1]
for case in cases:
    if case == 0:
        parameters = '{}network_params_microcircuit.yaml'.format(fixture_path)
        regime = 'noise_driven'
    elif case == 1:
        parameters = '{}minimal_negative.yaml'.format(fixture_path)
        regime = 'negative_firing_rate'
    elif case == 2:
        parameters = '{}small_network.yaml'.format(fixture_path)
        regime = 'mean_driven'
    else:
        print('Case not defined! Choose existing case, '
              'otherwise nothing happens!')

    network = lmt.Network(parameters, ('{}analysis_params_test.yaml'
                                       ).format(fixture_path))

    network.network_params['regime'] = regime

    network.working_point()

    network.transfer_function(method='shift')
    network.results['tf_shift'] = network.results.pop('transfer_function')
    network.transfer_function(method='taylor')
    network.results['tf_taylor'] = network.results['transfer_function']

    original_delay_dist = network.network_params['delay_dist']
    network.network_params['delay_dist'] = 'none'
    network.delay_dist_matrix()
    dd_none = network.results['delay_dist']
    network.network_params['delay_dist'] = 'truncated_gaussian'
    network.delay_dist_matrix()
    dd_truncated_gaussian = network.results.pop('delay_dist')
    network.network_params['delay_dist'] = 'gaussian'
    network.delay_dist_matrix()
    dd_gaussian = network.results.pop('delay_dist')
    network.results['delay_dist_all'] = [dd_none.magnitude,
                                         dd_truncated_gaussian.magnitude,
                                         dd_gaussian.magnitude]*dd_none.units
    network.network_params['delay_dist'] = original_delay_dist

    omega = network.analysis_params['omega']
    network.sensitivity_measure(omega)
    network.transfer_function(omega)
    network.delay_dist_matrix()
    network.power_spectra()

    network.eigenvalue_spectra('MH')
    network.eigenvalue_spectra('prop')
    if regime != 'negative_firing_rate':
        network.eigenvalue_spectra('prop_inv')

    network.r_eigenvec_spectra('MH')
    network.r_eigenvec_spectra('prop')
    if regime != 'negative_firing_rate':
        network.r_eigenvec_spectra('prop_inv')

    network.l_eigenvec_spectra('MH')
    network.l_eigenvec_spectra('prop')
    if regime != 'negative_firing_rate':
        network.l_eigenvec_spectra('prop_inv')

    network.results['additional_rates_for_fixed_input'] = (
        network.additional_rates_for_fixed_input(
            network.network_params['mean_input_set'],
            network.network_params['std_input_set']))

    params = network.network_params
    #
    # fixtures = dict(params, **network.results)
    # fixtures = dict(fixtures, **network.analysis_params)
    #
    # fixtures = lmt.input_output.quantities_to_val_unit(fixtures)

    # for k, v in fixtures.items():
    #     try:
    #         fixtures[k] = v.tolist()
    #     except AttributeError:
    #         try:
    #             fixtures[k]['val'] = v['val'].tolist()
    #         except AttributeError:
    #             pass
    #         except TypeError:
    #             pass
    network.save(file_name='{}{}_regime.h5'.format(fixture_path, regime))
    # with open('{}{}_regime.yaml'.format(fixture_path, regime), 'w') as file:
    #     yaml.dump(fixtures, file)
