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

fixtures_input_path = 'tests/unit/fixtures/input/'
fixtures_output_path = 'tests/unit/fixtures/output/'



def fixtures_Phi():
    
    function_name = 'Phi'
    
    input_file = fixtures_input_path + function_name + '.npy'
    output_file = fixtures_output_path + function_name + '.npy'
    
    lp = -5
    hp = 1.5
    test_inputs = np.concatenate([-np.logspace(hp, lp),[0],np.logspace(lp, hp)])
    
    np.save(input_file, test_inputs)
    
    results = []
    for test_input in test_inputs:
        results.append(Phi(test_input))
    
    np.save(output_file, results)

    

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
    
    pcfu_results = []
    for z, x in zip(zs, xs):
        pcfu_results.append(mpmath.pcfu(z, -x))
        
    np.savez(input_file, zs=zs, xs=xs, pcfu=pcfu_results)

    results = []
    for z, x  in zip(zs, xs):
        results.append(Psi(z, x))
    
    np.save(output_file, results)
    
    
def fixtures_d_Psi():
    
    function_name = 'd_Psi'
    
    input_file = fixtures_input_path + function_name + '.npz'
    output_file = fixtures_output_path + function_name + '.npy'
    
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
    
    
def fixtures_d_2_Psi():
    
    function_name = 'd_2_Psi'
    
    input_file = fixtures_input_path + function_name + '.npz'
    output_file = fixtures_output_path + function_name + '.npy'
    
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
    
    
def fixtures_p_hat_boxcar():
    
    function_name = 'p_hat_boxcar'
    
    input_file = fixtures_input_path + function_name + '.npz'
    output_file = fixtures_output_path + function_name + '.npy'
    
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
    
    # fixtures_Phi()
    # fixtures_Phi_prime_mu()
    # fixtures_Psi()
    # fixtures_d_Psi()
    # fixtures_d_2_Psi()
    fixtures_p_hat_boxcar()
