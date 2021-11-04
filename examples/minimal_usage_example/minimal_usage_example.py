"""
Minimal usage example
=====================
"""

import nnmt
ureg = nnmt.ureg

# instantiate network
network = nnmt.models.Microcircuit(
    'network_params_microcircuit.yaml',
    'analysis_params.yaml')

# calculate working point
wp = nnmt.lif.exp.working_point(network)

# print results
print('Working point:')
print(f"mean input: {wp['mean_input']}")
print(f"std input: {wp['std_input']}")
print(f"firing rates: {wp['firing_rates']}")

# save results to h5 file
network.save('microcircuit_working_point.h5')
