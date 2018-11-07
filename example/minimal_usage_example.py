# Minimal working example

import lif_meanfield_tools as lmt
ureg = lmt.ureg

# instantiate network
network = lmt.Network(network_params='network_params_microcircuit.yaml',
                      analysis_params='analysis_params.yaml')

# # calculate working point
working_point = network.working_point()

#print results
print('Working point:')
print('mean input: {}'.format(working_point['mean_input']))
print('std input: {}'.format(working_point['std_input']))
print('firing rates: {}'.format(working_point['firing_rates']))

# print(network.transfer_function())
print(network.transfer_function(10*ureg.Hz))

# save results
network.save()
