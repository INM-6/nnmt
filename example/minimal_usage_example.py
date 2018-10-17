# Minimal working example

import lif_meanfield_tools as lmt
from lif_meanfield_tools import ureg

# instantiate network
network = lmt.Network(network_params='network_params_microcircuit.yaml',
                      analysis_params='analysis_params.yaml')

# # calculate working point
working_point = network.working_point()

#print results
print('Working point:')
print('mean: {}'.format(working_point['mu']))
print('std: {}'.format(working_point['sigma']))
print('firing rates: {}'.format(working_point['firing_rates']))

# save results
network.save()
