.. _sec_lif:

===
LIF
===

Here you find tools for the leaky integrate-and-fire (LIF) neuron. The module
is divided into submodules corresponding to synapse types (see theory section
below).

.. autosummary::
  :toctree: _autosummary
  :template: custom-submodule-template.rst

  nnmt.lif.delta
  nnmt.lif.exp
  nnmt.lif._general

******
Theory
******

The differential equation describing the membrane potential :math:`V` of the
leaky integrate-and-fire (LIF) neuron is given by

.. math::

    \tau_\mathrm{m} \frac{\mathrm{d}V}{\mathrm{d}t} =
    -V + I_{\mathrm{syn}}(V,t) + I_\mathrm{ext}(t) \quad ,

with membrane time constant :math:`\tau_\mathrm{m}`,
synaptic current :math:`I_{\mathrm{syn}}(V,t)`,
and external current :math:`I_{\mathrm{ext}}(t)`. Once the membrane voltage
reaches a threshold :math:`V_\Theta`, it is reset to the reset potential
:math:`V_0` and a spike is emitted.

See :cite:t:`fourcaud2002` for more details.

Delta synapses
==============

For instantaneous or delta synapses the synaptic current is given by

.. math::

    I_{\mathrm{syn}}(t) = \sum_{i=1}^{N_\mathrm{s}} J_i
    \sum_k \delta(t-t_i^k) \tau_\mathrm{m} \quad ,

where the first sum runs over all :math:`N_\mathrm{s}` synapses and the second
sum over all presynaptic spikes of each synapse. :math:`t_i^k` is the time at
which spike :math:`k` arrives at synapse :math:`i`. :math:`J_i` is the synaptic
efficacy or weight of synapse :math:`i`.

Exponential synapses
====================

For synapses with instantaneous jump and exponential decay with time constant
:math:`\tau_\mathrm{s}`, or just exponential synapses, the synaptic current is
given by

.. math::

    \tau_\mathrm{s}\frac{\mathrm{d} I_{\mathrm{syn}}}{\mathrm{d} t}
    = -I_\mathrm{syn}(t)
    + \sum_{i=1}^{N_\mathrm{s}} J_i \sum_k \delta(t-t_i^k) \tau_\mathrm{m} \quad .

*********
Variables
*********

Here you find how variables of LIF neurons are named in NNMT:

.. list-table::
   :widths: 50 25 25
   :header-rows: 1

   * - Quantity
     - Symbol
     - Variable name
   * - Synaptic delay matrix
     - :math:`\boldsymbol{D}`
     - ``D``
   * - Synaptic weight matrix
     - :math:`\boldsymbol{J}`
     - ``J``
   * - External synaptic weight matrix
     - :math:`\boldsymbol{J}_\mathrm{ext}`
     - ``J_ext``
   * - Indegree matrix
     - :math:`\boldsymbol{K}`
     - ``K``
   * - External indegree matrix
     - :math:`\boldsymbol{K}_\mathrm{ext}`
     - ``K_ext``
   * - Number of neurons
     - :math:`\boldsymbol{N}`
     - ``N``
   * - Reset voltage
     - :math:`V_0`
     - ``V_0``
   * - Threshold voltage
     - :math:`V_\Theta`
     - ``V_th``
   * - Mean of synaptic input
     - :math:`\boldsymbol{\mu}`
     - ``mu``
   * - Population rates
     - :math:`\boldsymbol{\nu}`
     - ``nu``
   * - External population rates
     - :math:`\boldsymbol{\nu}_\mathrm{ext}`
     - ``nu_ext``
   * - Angular frequencies
     - :math:`\omega`
     - ``omegas``
   * - Standard deviation of synaptic input
     - :math:`\boldsymbol{\sigma}`
     - ``sigma``
   * - Membrane time constant
     - :math:`\tau_\mathrm{m}`
     - ``tau_m``
   * - Refractory time
     - :math:`\tau_\mathrm{r}`
     - ``tau_r``
   * - Synaptic time constant
     - :math:`\tau_\mathrm{s}`
     - ``tau_s``
