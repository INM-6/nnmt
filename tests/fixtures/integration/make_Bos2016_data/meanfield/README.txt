Package calculating stationary and dynamical properties of networks 
composed of leaky integrate-and-fire neurons connected with exponentially 
decaying synapses.
=========================================================================


Dependencies
------------

libmathlib2-gfortran
libmathlib2-dev
-> available via sudo apt-get install

h5py_wrapper
-> available on https://github.com/INM-6/h5py_wrapper


Documentation
-------------

The main class is Circuit() in circuit.py. Depending on the chosen 
analysis_type (None, 'stationary', 'dynamical'), an instantiation of 
Circuit() offers function to calculate the stationary and dynamical 
properties of the circuit. For example:

- firing rates (Brunel & Hakim 1999, Fourcoud & Brunel 2002)
- transfer functions (Schuecker 2015)
- power spectra and their anatomical origin (Bos 2015, Schuecker 2015)

By default the analysis_type is set to 'dynamical' and the transfer 
function is calculated for the all frequencies which might be time 
consuming. The paramters of the circuit are specified in params_circuit.py. 
All parameters including the analysis_type can be altered after the 
circuit has been initialised using the function alter_default_params().


References
----------

- Brunel N, Hakim V (1999) Fast global oscillations in networks of 
  integrate-and-fire neurons with low firing rates. Neural Comput. 
  11:1621–1671.
- Fourcaud N, Brunel N (2002) Dynamics of the firing probability of noisy 
  integrate-and-fire neurons. Neural Comput. 14:2057–2110.
- Schuecker J, Diesmann M, Helias M. Modulated escape from a metastable 
  state driven by colored noise. Phys Rev E. 2015 Nov;92:052119. 
  Available from: http://link.aps.org/doi/10.1103/PhysRevE.92.052119.
- Bos H, Diesmann M, Helias M (2015) Identifying anatomical origins of 
  coexisting oscillations in the cortical microcircuit 
  arXiv:1510.00642 [q-bio.NC]


Examples
--------

# Calculation of firing rates

import circuit

circ = circuit.Circuit('microcircuit', analysis_type='stationary')
print 'firing rates', circ.th_rates

# Calculation of population rate spectra

import matplotlib.pyplot as plt
import numpy as np
import circuit

dic = {'dsd': 1.0, 'delay_dist': 'truncated_gaussian'}
circ = circuit.Circuit('microcircuit', dic)
freqs, power = circ.create_power_spectra()

plt.figure()
for i in range(8):
    plt.plot(freqs, np.sqrt(power[i]), label=circ.populations[i])
plt.yscale('log')
plt.legend()
plt.show()
