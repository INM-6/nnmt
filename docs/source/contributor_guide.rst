=================
Contributor guide
=================

Welcome! You are now entering the contributor guide. This place is for people
interested in the details needed to contribute to NNMT. We want this toolbox to
be a place for the Neuroscientific community to collect analytical methods for
neuronal network model analysis. So we are glad that you made your way here and
hope that you will find what you are looking for. Good luck and have a lot of
fun!

Contributions to our `GitHub repository <https://github.com/INM-6/nnmt>`_ can
be made via the GitHub fork and pull request workflow.

Any suggestions or problems should be reported as
`issues on GitHub <https://github.com/INM-6/nnmt/issues>`_.

***********************************
What belongs here and what does not
***********************************

Here we should give a detailed explanation of what we think should be part of
NNMT and what should not. But, as the toolbox is still in its infancy, this is
rather difficult to define. We do not know yet, how the toolbox might - or
should - develop in the future. At the moment we only can state the general
purpose of this package:

NNMT is a Python toolbox for collecting analytical methods for neuronal network
model analysis.

******************************************
Structure of toolbox and design principles
******************************************

.. image:: images/package_structure.png
  :width: 600
  :alt: Sketch of structure of python package

Structure
=========

NNMT is divided into tools and methods and has a flexible, modular structure.
The best description of the ideas behind this can be found in our paper:
`NNMT: A mean-field toolbox for spiking neuronal network model analysis <add missing link>`_.

Design principles
=================

These are some thoughts that we had when we wrote the core package. They should
be followed when writing new code for the toolbox:

- **All calculations are to be done in SI units.** We do not use Python
  quantity packages like ``pint`` or ``quantitites`` inside the actual
  calculations because this often causes problems in combination with special
  functions (e.g. ``erf`` or ``zetac`` from SciPy). Although we do use
  ``pint`` for converting parameters including units from yaml files to
  dictionaries. For more detail see the :ref:`models section <subsec models>`.
- **Resuse as much code as possible.** If two functions in
  different submodules (e.g. in ``lif.exp`` and ``lif.delta``) use the same
  function, the function should be put into a higher module at a higher level.
  In the ``lif`` module we introduced the ``_static`` module which serves this
  purpose. Keep in mind that if two modules that need a similar function are
  not both part of the same submodule, it might be sensible to combine them in
  a new submodule.
- **The package's structure is supposed to be adapted in a flexible,
  non-dogmatic way.** If the canonical split into neuron type, synapse type
  doesn't fit, feel free to adjust the submodule structure accordingly. An
  inspiration to us was the submodule structure of SciPy, which (at least
  it seemed so to us) is rather free and fitted to the needs at hand.

Tools
=====

Tools are **Python functions** and constitute the core of NNMT. They actually
perform the calculations.

We decided to **sort them into different submodules**. Originally, starting off
with tools for LIF neurons, we thought the most sensible split is according to
neuron type (e.g. LIF, binary, etc.) and then, if required, another split
according to synapse type (e.g. delta, exponential). But analytical theories of
neuronal network models are very versatile. Therefore other ways of sorting the
tools might be more appropriate for different tools.

It is vital that all tools have **meaningful names** and
**comprehensive docstrings** (see :ref:`documentation section <subsec docs>`
for more details).

If you make any well-thought-out decisions in the implementation of a tool, for
example for optimization purposes, you need to **write comments** that clearly
state the reasons for you to do so. Otherwise, someone else might come across
your lines of code a few years later and change it, because it looked
unnecessarily cumbersome at first sight, thereby destroying all your precious
efforts.

_Tools
******

Tools with an underscore are where the job is done. Underscored tools should:

- **get** all **parameters** needed for a calculation **directly as**
  **arguments**.
- **perform the calculations**.
- **assert** that all arguments have **valid values**. For example, they need
- to check whether parameters that only should be positive are negative.
- **raise warnings if valid parameter regime is left**. For example if the
  assumptions made in the underlying theory are not fulfilled by the
  parameters.
- **raise errors if return values are meaningless**. For example if negative
  rates would be returned.

Wrappers
********

To make an underscored tool compatible with the convience layer, a.k.a. models,
it gets a wrapper withouth an underscore. The non underscored wrappers should:

- **expect an ``nnmt.model`` as argument**.
- **check** that all **parameters and results needed are stored in the model**.
- invoces the _cache function to **store the results**.

.. _subsec models:

Models
======

- models derive from the network class
- each model is supposed to calculate necessary parameters when instantiated
- therefore you can add methods and invoke them in the __init__ method.
- they might read in yaml files, specifying the model parameters including
  units.
- when instantiated the parameters are loaded, converted to SI units and then
  stripped off units saved in input_units dict
- result units stored in results_units
- most important: results hash dict vs results dict


Utils
=====

- most important: cache and how it works

*****
Tests
*****

- explained in detail in :ref:`test section <mytests>`


.. _subsec docs:

*************
Documentation
*************

- mostly automatic using sphinx and rst files
- source vs build
- conf.py
- index.rst
- make clean, make html
- link to sphinx documentation
- Need to list functions in module docstring
- follow numpy standard (link)
- in wrapper or underscored function?
- if in wrapper, you need to list the network params, analysis params and
  results needed