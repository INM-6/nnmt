# Minimal working example
import lif_meanfield_tools as lmt
ureg = lmt.ureg

# instantiate network
network = lmt.models.Microcircuit('network_params_microcircuit.yaml',
                                    'analysis_params.yaml')

# calculate working point
wp = lmt.lif.exp.working_point(network)

# print results
print('Working point:')
print(f"mean input: {wp['mean_input']}")
print(f"std input: {wp['std_input']}")
print(f"firing rates: {wp['firing_rates']}")

# calculate transfer function
tf = lmt.lif.exp.transfer_function(network)
print(tf)

network.save('test.h5')
