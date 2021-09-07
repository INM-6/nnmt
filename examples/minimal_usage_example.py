"""
Minimal working example
=======================
"""

import nnmt
ureg = nnmt.ureg

# instantiate network
network = nnmt.models.Microcircuit(
    'parameters/network_params_microcircuit.yaml',
    'parameters/analysis_params.yaml')

# calculate working point
wp = nnmt.lif.exp.working_point(network)

# print results
print('Working point:')
print(f"mean input: {wp['mean_input']}")
print(f"std input: {wp['std_input']}")
print(f"firing rates: {wp['firing_rates']}")

# calculate transfer function
tf = nnmt.lif.exp.transfer_function(network)
print(tf)

network.save('temp/minimal_usage_example.h5')
