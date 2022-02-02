.. _sec_models:

======
Models
======

Here you find all the models currently available in NNMT.

Models in NNMT basically are containers for network parameters, analysis
parameters, and results. They come with convenience routines for changing
parameters, saving, and loading results. They can be instantiated using
``yaml`` files or dictionaries, and they typically require network parameters
and analysis parameters as arguments.

Please read the :ref:`overview <sec_overview>` for more details.

*************
Network class
*************

This is the parent class all other network models inherit from. It defines the
attributes NNMT tools assume to find and defines methods for changing
parameters, saving, and loading results.

.. autosummary::
  :toctree: _autosummary
  :template: custom-class-template.rst
  :recursive:

  nnmt.models.Network

**************************
Implemented network models
**************************

These are network models derived from the generic :class:`nnmt.models.Network`
class. They define how parameter files are read in and how dependent network
and analysis parameters are calculated from the parameter files.

.. autosummary::
  :toctree: _autosummary
  :template: custom-class-template.rst
  :recursive:

  nnmt.models.Plain
  nnmt.models.Basic
  nnmt.models.Microcircuit

********************
Yaml parameter files
********************

Here we explain how to set up a yaml parameter file for instantiating a model.

Parameters are defined in a dictionary-like manner using colons and
Python-style indentation to indicate nesting. Elements of lists are preceded by
hyphens ``-``, and arrays can be expressed as nested lists.

You can either define parameters with units using the keys ``val`` and
``unit``, or define unitless variables without any key.

Which parameters you need to define depends on the model you want to use and is
indicated in the respective model's docstring.

The following code snippet contains examples of structures for defining
parameters:

.. code:: yaml

  <parameter>:
    val: <value>
    unit: <unit>

  <parameter_list>:
    val:
      - <value1>
      - <value2>
      - <value3>
    unit: <unit>

  <unitless_parameter>: <value>

  <unitless_parameter_list>:
    - <value1>
    - <value2>
    - <value3>

  <unitless_parameter_array>:
    - - <value11>
      - <value12>
    - - <value21>
      - <value22>

Yaml files for microcircuit model used in
:ref:`Power spectra and sensitivity measure in microcircuit model (Bos 2016) example <example_bos_2016>`:

- Network parameter file  :download:`network_params.yaml <../../tests/fixtures/integration/config/Bos2016_network_params.yaml>`
- Analysis parameter file :download:`analysis_params.yaml <../../tests/fixtures/integration/config/Bos2016_analysis_params.yaml>`.
