.. _sec_tools:

=====
Tools
=====

Here you find all of the tools currently in the toolbox, sorted by submodule.

The tools, which are the implementations of the contained analytical methods,
are the core of NNMT. There are underscored _tools that expect all required
parameters direcly as arguments, and respective wrapper tools that expect a
:ref:`model <sec_models>` as argument.

Please read the :ref:`overview <sec_overview>` for more details.

******************************
Approximations and assumptions
******************************

Analytical analyses of neuronal netoworks almost always rely on approximations
and assumptions. If the network analyzed with a mean-field based tool does not
fulfill the tool's requirements, it cannot provide reliable results. These
restrictions should be documented in the docstrings of the respective tools.
Here, we explain a few important terms that appear in that context.

- **Diffusion approximation**: If a neuron receives Poissonian uncorrelated
  input spike trains and the contribution of a single syanptic connection is
  small compared to the distance between reset and threshold
  :math:`w \ll \left(V_\Theta - V_0\right)`, the random input can be
  approximated by Gaussian white noise with mean :math:`\mu` and noise
  intensity :math:`\sigma^2` :cite:p:`tuckwell1988,amit1991`. This
  approximation does not hold if the network features highly correlated
  activity or receives strong external input common to many neurons.
- **Linear response theory**: Studies how populations of neurons
  in the stationary state respond to weak external input, ignoring non-linear
  interactions. Linear response theory cannot explain higher order effects like
  the occurence of higher harmonics.
- **Fast/slow synaptic regime**: Parameter regime in which the synaptic time
  constant :math:`\tau_\mathrm{s}` is much shorter/longer than the membrane
  time constant :math:`\tau_\mathrm{m}`.

******************
Network properties
******************

.. toctree::
   :maxdepth: 1

   network_properties

********************************
Leaky integrate-and-fire neurons
********************************

.. toctree::
   :maxdepth: 1

   lif

*****************************
Spatially structured networks
*****************************

.. toctree::
   :maxdepth: 1

   spatial

*************************
Linear stability analysis
*************************

.. toctree::
   :maxdepth: 1

   linear_stability
