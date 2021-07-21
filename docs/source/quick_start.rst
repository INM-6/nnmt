===========
Quickstart
===========

First you create a network model by using one of the predefined models in
``nnmt.models``

.. code:: python

    network = nnmt.models.Microcircuit('network_params.yaml',
                                       'analysis_params.yaml')
                                       
or by defining one yourself, using the standard template
``nnmt.models.Network``:

.. code:: python

    network = nnmt.models.Network('network_params.yaml',
                                  'analysis_params.yaml')
    # synaptic weight in mV
    w = 1.2e-3
    # weight matrix
    network['J'] = np.array([[1, -4],
                             [1, -4]]) * w
    # inputs matrix
    network['K'] = np.array([[3000, 1000],
                             [3000, 1000]])
                             
Here we have defined some parameters using `yaml` files. Those contain the
paramters listed in the following format:

.. code:: yaml

    tau_m:
      val: 10
      unit: ms
      
    V_th:
      val:
        - -50
        - -60
      unit: mV
    
A model in NNMT basically is a container for network parameters,
analysis parameters, and calculated results. You can use a model`s methods
to save and load results.

Once you have defined your model, you can use the tools to calculate estimates
of quantities like firing rates, power spectra or use other more elaborated
methods you can find in this documentation. To do this, you simply choose a
tool and apply it to the model:

.. code:: python

    firing_rates = nnmt.lif.exp.firing_rates(network)
    
If you are missing some parameters for applying the tool you would like to use,
you will receive an error message telling you, which parameters you need to
define:

.. code:: console

    RuntimeError: You are missing 'tau_m' for calculating the firing rate!
    Have a look into the documentation for more details on 'lif' parameters.
    
Sometimes, before you can calculate a quantity, you first have to calculate
something else. In that case, you will receive an error message as well. Here

.. code:: python
    power_spectra = nnmt.lif.exp.power_spectra(network)
    
will lead to the error

.. code:: console
    
    RuntimeError: You first need to calculate the 'lif.exp.effective_connectivity'.

because calculating the power spectra requires calculating the effective
connectivity first.

If you do not want to use a model, you can calculate a quantity, using the
corresponding underscore functions, which takes the required parameters
directly:

.. code:: python

    firing_rates = nnmt.lif.exp._firing_rates(J, K, V_0_rel, V_th_rel,
                                              tau_m, tau_r, tau_s,
                                              J_ext, K_ext, nu_ext)
