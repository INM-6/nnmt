.. _sec_release_notes:

=============
Release notes
=============

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