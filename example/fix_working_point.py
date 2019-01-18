"""
fix_working_point.py

Example for creating a new network with the mean and standard deviation of the
input fixed by setting additional external Poisson input rates.

Author: Johanna Senk
"""
import lif_meanfield_tools as lmt
ureg = lmt.ureg

default_nw = lmt.Network(network_params='network_params_microcircuit.yaml',
                         analysis_params='analysis_params.yaml')

working_point = default_nw.working_point()
print('Working point of network with default parameters:')
print('mean input: {}'.format(working_point['mean_input']))
print('std input: {}'.format(working_point['std_input']))
print('firing rates: {}'.format(working_point['firing_rates']))
print('')

# mean and standard deviation of input
mu_set = working_point['mean_input']
mu_set[0] *= 0.9 # modification to one population
sigma_set = working_point['std_input']
new_nu_e_ext, new_nu_i_ext = \
    default_nw.additional_rates_for_fixed_input(mu_set, sigma_set)

print('New additional external rates:')
print('new_nu_e_ext: {}'.format(new_nu_e_ext))
print('new_nu_i_ext: {}'.format(new_nu_i_ext))
print('')

new_nw = lmt.Network(network_params='network_params_microcircuit.yaml',
                     analysis_params='analysis_params.yaml',
                     new_network_params={'nu_e_ext': new_nu_e_ext,
                                         'nu_i_ext': new_nu_i_ext})

working_point = new_nw.working_point()
print('Working point of network with new parameters:')
print('mean input: {}'.format(working_point['mean_input']))
print('std input: {}'.format(working_point['std_input']))
print('firing rates: {}'.format(working_point['firing_rates']))

new_nw.save()
