.. _sec_models:

======
Models
======

Network objects in NNMT basically are containers for network parameters,
analysis parameters, and results. They come with convenient saving and loading
routines and allow defining how to convert your parameter files (we mostly use
``yaml`` files, but you can use whatever suits you the best) to network
parameters.

Network parameters are all parameters that describe properties of the network
itself, like for example the number of neurons in each population, or the
membrane time constants.

Analysis parameters are all parameters that do not describe properties of the
network, but need to be defined in order to calculate quantities of interest.
For example, one needs to define the frequencies for which to calculate the
power spectra.

Using networks can make calculating quantities much easier, because many
quantities rely on previous results. When you calculate a quantity using a
network, the results are always stored, so other functions can grab those
results easily.

*************
Network class
*************

This is the parent class, all other network models inherit from. It defines the
attributes NNMT functions assume to find and defines saving and loading
methods, as well as some convenience methods.

.. autosummary::
  :toctree: _autosummary
  :template: custom-class-template.rst
  :recursive:

  nnmt.models.Network

**************************
Implemented network models
**************************

These are network models derived from the generic ``Network`` object. They
define how parameter files are read in and how network parameters and analysis
parameters are calculated.

.. autosummary::
  :toctree: _autosummary
  :template: custom-class-template.rst
  :recursive:

  nnmt.models.Basic
  nnmt.models.Microcircuit
