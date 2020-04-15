""" 
Only run this script, if you are sure that all functions in aux_calcs.py are correct!
"""

import numpy as np

from mpmath import pcfu

# temporarily add local version of lif_meanfield_tools to pythonpath 
# this is necessary to create the data using the local version and not the 
# installed module (important for debugging)
import sys
sys.path.insert(1, './')
from lif_meanfield_tools.aux_calcs import *

from lif_meanfield_tools import ureg


class Fixtures():
    
    def __init__(self, input_path, output_path):
        
        self.input_path = input_path
        self.output_path = output_path
        
        # noise driven regime mu < V_th
        # parameters taken from microcircuit example
        mus = [3.30031035, 7.02709379, 7.18151477, 9.18259078] * ureg.mV
        sigmas = [6.1901737, 5.11420662, 5.96478947, 4.89397196] * ureg.mV

        tau_m = 10. * ureg.ms
        tau_s = 0.5 * ureg.ms
        tau_r = 2 * ureg.ms
        V_th_rel = 15 * ureg.mV
        V_0_rel = 0 * ureg.mV

        tau_ms = np.repeat(tau_m, len(mus))
        tau_ss = np.repeat(tau_s, len(mus))
        tau_rs = np.repeat(tau_r, len(mus))
        V_th_rels = np.repeat(V_th_rel, len(mus))
        V_0_rels = np.repeat(V_0_rel, len(mus))

        self.parameters_noise_driven_regime = dict(mu=mus, sigma=sigmas, 
                                                   tau_m=tau_ms, tau_s=tau_ss,
                                                   tau_r=tau_rs, 
                                                   V_th_rel=V_th_rels,
                                                   V_0_rel=V_0_rels)

        # regime in which negative firing rates occured once
        # parameters taken from circuit in which lmt returned negative rates
        mus = [-12.88765852, -21.41462729, 6.76113423, -4.69428276] * ureg.mV
        sigmas = [9.26667293, 10.42112985, 4.56041, 13.51676476] * ureg.mV

        tau_m = 20 * ureg.ms
        tau_s = 2 * ureg.ms
        tau_r = 2 * ureg.ms
        V_th_rel = 15 * ureg.mV
        V_0_rel = 0 * ureg.mV

        tau_ms = np.repeat(tau_m, len(mus))
        tau_ss = np.repeat(tau_s, len(mus))
        tau_rs = np.repeat(tau_r, len(mus))
        V_th_rels = np.repeat(V_th_rel, len(mus))
        V_0_rels = np.repeat(V_0_rel, len(mus))

        self.parameters_negative_firing_rate_regime = dict(mu=mus, sigma=sigmas, 
                                                           tau_m=tau_ms, tau_s=tau_ss,
                                                           tau_r=tau_rs, 
                                                           V_th_rel=V_th_rels,
                                                           V_0_rel=V_0_rels)

        # mean driven regime mu > V_th
        # parameters taken from adjusted microcircuit example
        mus =  [741.89455754, 21.24112709, 35.86521795, 40.69297877, 651.19761921] * ureg.mV
        sigmas = [39.3139564, 6.17632725, 9.79196704, 10.64437979, 37.84928217] * ureg.mV

        tau_m = 20. * ureg.ms
        tau_s = 0.5 * ureg.ms
        tau_r = 0.5 * ureg.ms
        V_th_rel = 20 * ureg.mV
        V_0_rel = 0 * ureg.mV

        tau_ms = np.repeat(tau_m, len(mus))
        tau_ss = np.repeat(tau_s, len(mus))
        tau_rs = np.repeat(tau_r, len(mus))
        V_th_rels = np.repeat(V_th_rel, len(mus))
        V_0_rels = np.repeat(V_0_rel, len(mus))

        self.parameters_mean_driven_regime =dict(mu=mus, sigma=sigmas, 
                                                 tau_m=tau_ms, tau_s=tau_ss,
                                                 tau_r=tau_rs, 
                                                 V_th_rel=V_th_rels,
                                                 V_0_rel=V_0_rels)

    def convert_dict_of_lists_to_array_of_dicts(self,d):
        list_of_dicts = [dict(zip(d, value)) for value in zip(*d.values())]
        array_of_dicts = np.array(list_of_dicts)
        return array_of_dicts
    
    def dict_without_key(self, d, key):
        temp = d.copy()
        temp.pop(key)
        return temp

    def siegert1(self):
        
        function_name = 'siegert1'
        
        # noise driven regime 
        inputs = self.dict_without_key(self.parameters_noise_driven_regime, 'tau_s')
        inputs = self.convert_dict_of_lists_to_array_of_dicts(inputs)   
        input_file = self.input_path + function_name + '_noise_driven_regime' + '.npy'
        np.save(input_file, inputs)
        
        # regime in which negative firing rates occured once
        inputs = self.dict_without_key(self.parameters_negative_firing_rate_regime, 'tau_s')
        inputs = self.convert_dict_of_lists_to_array_of_dicts(inputs)    
        input_file = self.input_path + function_name + '_negative_firing_rate_regime' + '.npy'
        np.save(input_file, inputs)        

    def siegert2(self):
        
        function_name = 'siegert2'

        # mean driven regime mu > V_th
        inputs = self.dict_without_key(self.parameters_mean_driven_regime, 'tau_s')
        inputs = self.convert_dict_of_lists_to_array_of_dicts(inputs)    
        input_file = self.input_path + function_name + '_mean_driven_regime' + '.npy'
        np.save(input_file, inputs)
        
    def nu0_fb433(self):
        self._colored_noise_firing_rate_functions('nu0_fb433')

    def nu0_fb(self):
        self._colored_noise_firing_rate_functions('nu0_fb')

    def _colored_noise_firing_rate_functions(self, function_name):
        
        # noise driven regime
        inputs = self.convert_dict_of_lists_to_array_of_dicts(self.parameters_noise_driven_regime)
        input_file = self.input_path + function_name + '_noise_driven_regime' + '.npy'
        np.save(input_file, inputs)
        
        # mean driven regime
        inputs = self.convert_dict_of_lists_to_array_of_dicts(self.parameters_mean_driven_regime)
        input_file = self.input_path + function_name + '_mean_driven_regime' + '.npy'
        np.save(input_file, inputs)
        
        # regime in which negative firing rate once occurred
        inputs = self.convert_dict_of_lists_to_array_of_dicts(self.parameters_negative_firing_rate_regime)
        input_file = self.input_path + function_name + '_negative_firing_rate_regime' + '.npy'
        np.save(input_file, inputs)
        
    def Phi(self):
        
        function_name = 'Phi'
        
        input_file = self.input_path + function_name + '.npy'
        output_file = self.output_path + function_name + '.npy'
        
        lp = -5
        hp = 1.5
        test_inputs = np.concatenate([-np.logspace(hp, lp),[0],np.logspace(lp, hp)])
        
        np.save(input_file, test_inputs)
        
        results = []
        for test_input in test_inputs:
            results.append(Phi(test_input))
        
        np.save(output_file, results)

    def Phi_prime_mu(self):
        
        function_name = 'Phi_prime_mu'
        
        input_file = self.input_path + function_name + '.npz'
        output_file = self.output_path + function_name + '.npy'
        
        lp = -5
        hp = 1.5
        steps = 20
        ss = np.concatenate([-np.logspace(hp, lp, steps),[0],np.logspace(lp, hp, steps)])
        sigmas = np.linspace(1, 100, 10)
        
        ss, sigmas = np.meshgrid(ss, sigmas)
        ss = ss.flatten()
        sigmas = sigmas.flatten()
        
        np.savez(input_file, ss=ss, sigmas=sigmas)
        
        results = []
        for s, sigma in zip(ss, sigmas):
            results.append(Phi_prime_mu(s, sigma))
        
        np.save(output_file, results)
        
    def Psi(self):
        
        function_name = 'Psi'
        
        input_file = self.input_path + function_name + '.npz'
        output_file = self.output_path + function_name + '.npy'
        
        z_range = np.concatenate([-np.logspace(2,-5, 4), [0], np.logspace(-5, 2, 4)])
        a, b = np.meshgrid(z_range, z_range)
        zs = a.flatten() + complex(0, 1)*b.flatten()
        xs = np.linspace(-10, 10, 8)
        
        zs, xs = np.meshgrid(zs, xs)
        zs = zs.flatten()
        xs = xs.flatten()
        
        pcfu_results = []
        for z, x in zip(zs, xs):
            pcfu_results.append(mpmath.pcfu(z, -x))
            
        np.savez(input_file, zs=zs, xs=xs, pcfu=pcfu_results)

        results = []
        for z, x  in zip(zs, xs):
            results.append(Psi(z, x))
        
        np.save(output_file, results)
        
    def d_Psi(self):
        
        function_name = 'd_Psi'
        
        input_file = self.input_path + function_name + '.npz'
        output_file = self.output_path + function_name + '.npy'
        
        z_range = np.concatenate([-np.logspace(2,-5, 4), [0], np.logspace(-5, 2, 4)])
        a, b = np.meshgrid(z_range, z_range)
        zs = a.flatten() + complex(0, 1)*b.flatten()
        xs = np.linspace(-10, 10, 8)
        
        zs, xs = np.meshgrid(zs, xs)
        zs = zs.flatten()
        xs = xs.flatten()
        
        psi_results = []
        for z, x in zip(zs, xs):
            psi_results.append(Psi(z + 1, x))
            
        np.savez(input_file, zs=zs, xs=xs, psi=psi_results)

        results = []
        for z, x  in zip(zs, xs):
            results.append(d_Psi(z, x))
        
        np.save(output_file, results)
        
    def d_2_Psi(self):
        
        function_name = 'd_2_Psi'
        
        input_file = self.input_path + function_name + '.npz'
        output_file = self.output_path + function_name + '.npy'
        
        z_range = np.concatenate([-np.logspace(2,-5, 4), [0], np.logspace(-5, 2, 4)])
        a, b = np.meshgrid(z_range, z_range)
        zs = a.flatten() + complex(0, 1)*b.flatten()
        xs = np.linspace(-10, 10, 8)
        
        zs, xs = np.meshgrid(zs, xs)
        zs = zs.flatten()
        xs = xs.flatten()
        
        psi_results = []
        for z, x in zip(zs, xs):
            psi_results.append(Psi(z + 2, x))
            
        np.savez(input_file, zs=zs, xs=xs, psi=psi_results)

        results = []
        for z, x  in zip(zs, xs):
            results.append(d_2_Psi(z, x))
        
        np.save(output_file, results)
        
    def p_hat_boxcar(self):
        
        function_name = 'p_hat_boxcar'
        
        input_file = self.input_path + function_name + '.npz'
        output_file = self.output_path + function_name + '.npy'
        
        lp = -5
        hp = 5
        steps = 20
        ks = np.concatenate([-np.logspace(hp, lp, steps),[0],np.logspace(lp, hp, steps)])

        widths = np.logspace(-5, 5)
        
        ks, widths = np.meshgrid(ks, widths)
        ks = ks.flatten()
        widths = widths.flatten()
        
        np.savez(input_file, ks=ks, widths=widths)
        
        results = []
        
        for k, width in zip(ks, widths):
            results.append(p_hat_boxcar(k, width))
            
        np.save(output_file, results)
    
    
if __name__ == '__main__':
    
    
    input_path = 'tests/unit/fixtures/input/'
    output_path = 'tests/unit/fixtures/output/'

    fixtures = Fixtures(input_path, output_path)
    fixtures.siegert1()
    fixtures.siegert2()
    fixtures.nu0_fb433()
    fixtures.nu0_fb()
    # fixtures.Phi()
    # fixtures.Phi_prime_mu()
    # fixtures.Psi()
    # fixtures.d_Psi()
    # fixtures.d_2_Psi()
    # fixtures.p_hat_boxcar()
