.. _sec_binary:

======
Binary
======

Here you find tools for binary neurons.

.. autosummary::
  :toctree: _autosummary
  :template: custom-submodule-template.rst

  nnmt.binary

******
Theory
******

For a description of how binary neurons and the respective network dynamics is
defined see :cite:t:`helias2014`.

*********
Variables
*********

Here you find how variables of binary neurons are named in NNMT:

.. list-table::
   :widths: 50 25 25
   :header-rows: 1

   * - Quantity
     - Symbol
     - Variable name
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
   * - Population rates
     - :math:`\boldsymbol{m}`
     - ``m``
   * - External population rates
     - :math:`\boldsymbol{m}_\mathrm{ext}`
     - ``m_ext``
   * - Mean of synaptic input
     - :math:`\boldsymbol{\mu}`
     - ``mu``
   * - Standard deviation of synaptic input
     - :math:`\boldsymbol{\sigma}`
     - ``sigma``
   * - Firing thresholds
     - :math:`\Theta`
     - ``theta``
