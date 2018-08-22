"""params_circuit.py: Summarizes parameters of neurons and the network in
dictionary. Each parameter set implements two functions. One specifying
all default parameter and one calculating the parameter that can be 
derived from the default parameter. The default parameter will be used
to create a hash when saving results.

Authors: Hannah Bos, Jannis Schuecker
"""

import numpy as np
import hashlib as hl
import h5py_wrapper.wrapper as h5


def get_data_microcircuit(new_params={}):
    """ Implements dictionary specifying all parameter of the microcircuit.

    Keyword arguments:
    new_params: dictionary, overwrites default parameter

    Output:
    params: dictionary with default and derived parameter
    param_keys: list of default parameter
    """
    params = {}

    params['populations'] = ['23E', '23I', '4E', '4I', 
                             '5E', '5I', '6E', '6I']
    # number of neurons in populations
    params['N'] = np.array([20683, 5834, 21915, 5479, 
                            4850, 1065, 14395, 2948])
        
    ### Neurons
    params['C'] = 250.0    # membrane capacitance in pF
    params['taum'] = 10.0  # membrane time constant in ms
    params['taur'] = 2.0   # refractory time in ms
    params['V0'] = -65.0   # reset potential in mV
    params['Vth'] = -50.0  # threshold of membrane potential in mV

    ### Synapses
    params['tauf'] = 0.5  # synaptic time constant in ms
    params['de'] = 1.5    # delay of excitatory connections in ms
    params['di'] = 0.75   # delay of inhibitory connections in ms
    # standard deviation of delay of excitatory connections in ms
    params['de_sd'] = params['de']*0.5 
    # standard deviation of delay of inhibitory connections in ms
    params['di_sd'] = params['di']*0.5
    # delay distribution, options: 'none', 'gaussian' (standard deviation 
    # is defined above), 'truncated gaussian' (standard deviation is 
    # defined above, truncation at zero)
    params['delay_dist'] = 'none'
    # PSC amplitude in pA
    params['w'] = 87.8*0.5

    ### Connectivity 
    # indegrees
    params['I'] = np.array([
        [2.19986486e+03, 1.07932007e+03, 9.79241261e+02, 4.67578108e+02, 
        1.59240826e+02, 0, 1.09819852e+02, 0],
        [2.99000583e+03, 8.60261056e+02, 7.03691807e+02, 2.89693864e+02,
         3.80735859e+02, 0, 6.05863901e+01, 0],
        [1.59875428e+02, 3.45225188e+01, 1.11717312e+03, 7.94596213e+02,
         3.26043349e+01, 3.19552818e-01, 6.67325211e+02, 0],
        [1.48097354e+03, 1.69432378e+01, 1.81302026e+03, 9.53325789e+02,
         1.60313926e+01, 0, 1.60812283e+03, 0],
        [2.18836598e+03, 3.74651134e+02, 1.13562969e+03, 3.13195876e+01,
         4.20770722e+02, 4.96471959e+02, 2.96694639e+02, 0],
        [1.16566761e+03, 1.59083568e+02, 5.70579343e+02, 1.20666667e+01,
         3.00095775e+02, 4.04172770e+02, 1.24332394e+02, 0],
        [3.25197985e+02, 3.86320250e+01, 4.67354637e+02, 9.17147621e+01,
         2.85670372e+02, 2.11899271e+01, 5.81635915e+02, 7.52183189e+02],
        [7.66905020e+02, 5.83683853e+00, 7.46380597e+01, 2.74016282e+00,
         1.36240841e+02, 8.55427408e+00, 9.79791723e+02, 4.59402985e+02]])
    # ratio of inhibitory to excitatory weights
    params['g']=4.0

    ### External input
    params['v_ext'] = 8.0 # in Hz
    # number of external neurons
    params['Next'] = np.array([1600,1500,2100,1900,2000,1900,2900,2100])

    ### Neural response
    # Transfer function is either calculated analytically ('analytical')
    # or approximated by an exponential ('empirical'). In the latter case
    # the time constants in response to an incoming impulse ('tau_impulse'), 
    # as well as the instantaneous rate jumps ('delta_f') have to be
    # specified.
    params['tf_mode'] = 'analytical'   
    if params['tf_mode'] == 'empirical': 
        params['tau_impulse'] = np.asarray([0.0 for i in range(8)])
        params['delta_f'] = np.asarray([0.0 for i in range(8)])
    # number of modes used when fast response time constants are calculated
    params['num_modes'] = 1

    # create list of parameter keys that are used to create hashes
    param_keys = params.keys()
    # Remove delay parameter from key list since they don't contribute
    # when calculating the working point and they are incorporated into
    # the transfer function after it has been read from file
    for element in ['de', 'di', 'de_sd', 'di_sd', 'delay_dist']:
        param_keys.remove(element)
       
    # file storing results
    params['datafile'] = 'results_microcircuit.h5'

    # update parameter dictionary with new parameters
    params.update(new_params)
    
    # calculate all dependent parameters
    params = get_dependend_params_microcircuit(params)

    return params, param_keys

def get_dependend_params_microcircuit(params):
    """Returns dictionary with parameter which can be derived from the
    default parameter.
    """
    # weight matrix, only calculated if not already specified in params
    if 'W' not in params:
        W = np.ones((8,8))*params['w']
        W[1:8:2] *= -params['g']
        W = np.transpose(W)
        # larger weight for L4E->L23E connections
        W[0][2] *= 2.0
        params['W'] = W

    # delay matrix
    D = np.ones((8,8))*params['de']
    D[1:8:2] = np.ones(8)*params['di']
    D = np.transpose(D)
    params['Delay'] = D

    # delay standard deviation matrix
    D = np.ones((8,8))*params['de_sd']
    D[1:8:2] = np.ones(8)*params['di_sd']
    D = np.transpose(D)
    params['Delay_sd'] = D
    
    return params

def create_hashes(params, param_keys):
    """Returns hash of values of parameters listed in param_keys."""
    label = ''
    for key in param_keys:
        value = params[key]
        if isinstance(value, (np.ndarray, np.generic)):
            label += value.tostring()
        else:
            label += str(value)
    return hl.md5(label).hexdigest()


