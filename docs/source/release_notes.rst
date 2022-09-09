.. _sec_release_notes:

=============
Release notes
=============

**********
NNMT 1.1.1
**********

- Add Hahne et al. 2017 and Layer et al. 2022 to delta firing rate references.

**********
NNMT 1.1.0
**********

- Move firing rate integration procedure used for lif neurons from
  ``nnmt.lif._general`` to ``nnmt._solvers``, where such general solving
  procedures are to be collected in the future.
- Add methods for binary neurons:
  - mean input
  - std input
  - mean activity
  - balanced threshold
- Add example comparing binary firing rates with simulation.
- Fix bug of fixture creation for lif neurons, which wouldn't create all
  fixtures on passing ``all``
- Move helper functions for lif fixture creation to own file.
- Add tests and fixture creation for binary neurons.

**********
NNMT 1.0.2
**********

- Fix calculation of mean input and std input for lif.exp. Previously ``tau_m``
  was multiplied with the firing rates before the dot product with the
  connectivity. However, as ``tau_m`` is representing the post-synaptic
  membrane time constant, it should be multiplied after performing the dot
  product.
- Fix explanation of ``tau_m`` in docstrings.
- Add new integration test for lif.exp._firing_rates with vector parameters.
- Fix docopt usage in fixture creation for unit and integration fixtures.


**********
NNMT 1.0.1
**********

- Deepcopy parameter dictionaries on instantiation of network model. Otherwise
  dictionary items can change unwantedly if netork parameters are changed.
- Add approximations and assumptions to docstrings.
- Add explanation of approximations to docs.
- Add table of LIF parameters and NNMT variables to docs.
- Fix description in docstrings for ``tau_s``.
- Fix typos in docstrings.
- Add new example of adjusting the low-gamma peak in the microcircuit model.
- Add pytest and pytest-mock to setup requirements, such that after pip
  installion the tests can be run.

**********
NNMT 1.0.0
**********

Initial release.