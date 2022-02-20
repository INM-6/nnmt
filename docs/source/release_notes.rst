.. _sec_release_notes:

=============
Release notes
=============

**********
NNMT 1.1.0
**********

- Move firing rate integration procedure used for lif neurons from
  ``nnmt.lif._general`` to ``nnmt._solvers``, where such general solving
  procedures are to be collected in the future.
- Add methods for binary neurons:
  - mean input
  - std input
  - firing rates
  - balanced threshold
- Add example comparing binary firing rates with simulation.
- Fix bug of fixture creation for lif neurons, which wouldn't create all
  fixtures on passing ``all``
- Move helper functions for lif fixture creation to own file.
- Add tests and fixture creation for binary neurons.

**********
NNMT 1.0.0
**********

Initial release.