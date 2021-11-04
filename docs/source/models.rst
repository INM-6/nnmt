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
