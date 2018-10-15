"""setup.py: Class handling the initialisation of all class variables 
available in Circuit and Analytics.

Authors: Hannah Bos, Jannis Schuecker
"""

import numpy as np
import params_circuit as pc


class Setup(object):
    """Class handling parameters and class variables of Circuit and 
    Analytics such that theses two classes share their variables at all
    times.
    Class variables of Circuit() are set by handing a Circuit() object 
    to the methods in Setup(), which return a dictionary with all new or 
    altered variables. Circuit() takes care of setting the class
    variables in Analytics().
    """
    def __init__(self):
        pass

    def get_default_params(self, params):
        """Returns dictionary with default parameter concerning the 
        calculations in Analytics(). 

        Arguments:
        params: dictionary, keys overwrite or extend default parameter
        """
        params_default = {'fmin': 0.1, 'fmax': 150., 'df': 1.0/(2*np.pi),
                          'to_file': True, 'from_file': True}
        params_default.update(params)
        return params_default

    def get_circuit_params(self, circ, new_params):
        """Returns dictionary with variables describing the circuit 
        parameter. The default parameter are specified in 
        params_circuit.py and overwritten by new_params.

        Arguments:
        circ: instance of Circuit() class
        new_params: parameter dictionary, used to overwrite default 
                    parameter specified in params_circuit.py
        label: string specifying the circuit parameter (listed in
               corresponding parameter dictionary in params_circuit.py)
        """
        new_vars = {}
        if circ.label == 'microcircuit':
            params, param_keys = pc.get_data_microcircuit(new_params)
            new_vars['param_keys'] = param_keys
            new_vars['param_hash'] = pc.create_hashes(params, param_keys) 
        else:
            raise RuntimeError('Parameter file missing for label.')
        new_vars['params'] = params
        return new_vars

    def get_params_for_analysis(self, circ):
        """Returns dictionary of parameter which are derived from 
        default analysis and circuit parameter.

        Arguments:
        circ: instance of Circuit() class
        """
        new_vars = {}
        w_min = 2*np.pi*circ.fmin
        w_max = 2*np.pi*circ.fmax
        dw = 2*np.pi*circ.df
        new_vars['omegas'] = np.arange(w_min, w_max, dw)
        return new_vars

    def get_params_for_power_spectrum(self, circ):
        """Returns dictionary of variables needed for calculation of
        the spectra.

        Arguments:
        circ: instance of Circuit() class
        """
        new_vars = {}
        if circ.params['tf_mode'] == 'analytical':
            new_vars['M'] = circ.params['I']*circ.params['W']
            new_vars['trans_func'] = circ.ana.create_transfer_function()
        else:
            for key in ['tau_impulse', 'delta_f']:
                new_vars[key] = circ.params[key]
            new_vars['H_df'] = circ.ana.create_H_df(new_vars, 'empirical')
            new_vars['M'] = circ.params['I']*circ.params['W']

        # copy of full connectivity (needed when connectivity is reduced)
        new_vars['M_full'] = new_vars['M']
        return new_vars

    def get_altered_circuit_params(self, circ, label):
        """Returns dictionary of parameter which are derived from 
        parameter associated to circuit.

        Arguments:
        circ: instance of Circuit() class
        label: string specifying the circuit
        """
        new_vars = {}
        if label == 'microcircuit':
            params = pc.get_dependend_params_microcircuit(circ.params)
        else:
            raise RuntimeError('Parameter file missing for label.')
        new_vars['param_hash'] = pc.create_hashes(params, circ.param_keys) 
        new_vars['params'] = params
        return new_vars

    def get_working_point(self, circ):
        """Returns dictionary of values determining the working point
        (the stationary properties) of the circuit.

        Arguments:
        circ: instance of Circuit() class
        """
        new_vars = {}
        th_rates = circ.ana.create_firing_rates()
        new_vars['mu'] = circ.ana.get_mean(th_rates)
        new_vars['var'] = circ.ana.get_variance(th_rates)
        new_vars['th_rates'] = th_rates
        return new_vars
