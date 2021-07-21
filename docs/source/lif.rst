LIF
===

Submodules
^^^^^^^^^^

The submodules correspond to the synapse type of the model used (see theory
section below).

.. autosummary::
  :toctree: _autosummary
  :template: custom-submodule-template.rst

  nnmt.lif.delta
  nnmt.lif.exp

Theory
^^^^^^

This is all about the leaky integrate and fire (LIF) neuron. The differential
equation describing the membrane potential :math:`V` in this model is given by

.. math::

    \tau_\mathrm{m} \frac{\mathrm{d}V}{\mathrm{d}t} =
    -V + I_{\mathrm{syn}}(V,t) + I_\mathrm{ext}(t) \quad ,
    
with membrane time constant :math:`\tau_m`,
synaptic current :math:`I_{\mathrm{syn}}(V,t)`,
and external current :math:`I_{\mathrm{ext}}(t)`. Once the membrane voltage
reaches a threshold :math:`V_\Theta`, it is reset to the reset potential
:math:`V_0`.

See :cite:t:`fourcaud2002` for more details.

Instantaneous synapses
""""""""""""""""""""""

For instantaneous or delta synapses the synaptic current is given by

.. math::

    I_{\mathrm{syn}}(t) = \sum_{i=1}^{N_\mathrm{s}} J_i
    \sum_k \delta(t-t_i^k) \tau_m \quad ,
    
where the first sum runs over all :math:`N_\mathrm{s}` synapses and the second
sum over all presynaptic spikes of each synapse. :math:`t_i^k` is the time at
which spike :math:`k` arrives at synapse :math:`i`. :math:`J_i` is the synaptic
efficacy or weight of synapse :math:`i`.
  
Synapses with instantaneous jump and exponential decay
""""""""""""""""""""""""""""""""""""""""""""""""""""""

For synapses with instantaneous jump and exponential decay or just exponential
synapses the synaptic current is given by

.. math::

    I_{\mathrm{syn}}(t) = -I_\mathrm{syn}(t) + \sum_{i=1}^{N_\mathrm{s}} J_i \sum_k \delta(t-t_i^k) \tau_m \quad .
  
