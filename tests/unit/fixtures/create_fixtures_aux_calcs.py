""" 
Only run this script, if you are sure that all functions in aux_calcs.py are correct!
"""

import numpy as np

from lif_meanfield_tools.aux_calcs import * 

fixtures_input_path = 'tests/unit/fixtures/input/'
fixtures_output_path = 'tests/unit/fixtures/output/'


def fixtures_Phi_prime_mu():
    
    function_name = 'Phi_prime_mu'
    
    input_file = fixtures_input_path + function_name + '.npz'
    output_file = fixtures_output_path + function_name + '.npy'
    
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
    

def fixtures_Psi():
    
    function_name = 'Psi'
    
    input_file = fixtures_input_path + function_name + '.npz'
    output_file = fixtures_output_path + function_name + '.npy'
    
    z_range = np.concatenate([-np.logspace(2,-5, 4), [0], np.logspace(-5, 2, 4)])
    a, b = np.meshgrid(z_range, z_range)
    zs = a.flatten() + complex(0, 1)*b.flatten()
    xs = np.linspace(-10, 10, 8)
    
    zs, xs = np.meshgrid(zs, xs)
    zs = zs.flatten()
    xs = xs.flatten()
    
    np.savez(input_file, zs=zs, xs=xs)

    results = []
    for z, x  in zip(zs, xs):
        results.append(Psi(z, x))
    
    np.save(output_file, results)
    
if __name__ == '__main__':
    
    fixtures_Phi_prime_mu()
    fixtures_Psi()
