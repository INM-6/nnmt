"""This is a minimal usage example.
Currently it's not working."""

import network as nw

# instantiate network
network = nw.Network(network_params='network_params_microcircuit.yaml',
                     analysis_params='analysis_params.yaml')

# calculate working point
working_point = network.working_point()

# print results
print('Working point:')
print('mean: {}, std: {}, firing rates: {}'.format(working_point['mu'],
                                                   working_point['sigma'],
                                                   working_point['firing_rates']))
