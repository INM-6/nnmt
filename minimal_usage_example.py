"""This is a minimal usage example.
Currently it's not working."""

import network as nw

# instantiate network
network = nw.Network(network_params='network_params.yaml', analysis_params='analysis_params.yaml', new_network_params={}, new_analysis_params={})

# calculate working point
working_point = network.working_point()

# print results
print('Working point:')
print('mean: {}, variance: {}, firing rates: {}'.format(working_point['mu'], working_point['var'], working_point['rates']))
