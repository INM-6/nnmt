.. _mytests:

=====
Tests
=====

We have an extensive test suite using the ``pytest`` framework. If you want to
run all the tests, you can simply do so by installing and activating the conda
environment specified in the provided ``environment.yaml`` file, and running:

.. code::

    pytest

from the root directory (the one containing ``tests/`` and ``nnmt/``). If you
want to be more specific, you can, for example, only run the unit tests:

.. code::

    pytest tests/unit/

Or just a single test:

.. code::

    pytest tests/unit/lif/test_exp.py::Test_firing_rate_shift::test_correct_output

See the `pytest documentation <https://docs.pytest.org/en/6.2.x/#>`_ for all
available options.

Note that ``pytest`` distinguishes between failures and errors:

- A failure occurs if a test did not run successfully.
- An error occurs if an exception happened outside of the test function, for
  example inside a fixture.

Additionally, tests can be skipped, and they can be marked as expected to fail
(xfail), which registers a fail if the respective test runs successfully.

Tests directory structure
=========================

.. code::

    tests/
    ├── checks.py
    ├── conftest.py
    ├── fixtures/
    ├── integration/
    ├── meta/
    └── unit/


Special files
*************

``checks.py`` is a collection of custom assert functions. If you need to write
a new one, add it here.

``conftest.py`` is a special ``pytest`` file, in which custom fixtures and
special ``pytest`` functions are defined. We, in particular, make use of the
``pytest_generate_tests`` function, which considerably simplifies complex
parametrizations of tests. For example, we use it to check the correct output
of several tools in different parameter regimes.

Fixtures
********

.. code::

    fixtures/
    ├── config.yaml
    ├── Snakefile
    ├── envs/
    ├── integration/
    │   ├── create_fixtures.py
    │   ├── config/
    │   ├── convert_Bos2016_data/
    │   ├── data/
    │   ├── make_Bos2016_data/
    │   └── make_Schuecker2015_data/
    └── unit/
        ├── config/
        ├── create/
        │   ├── lif_fixtures.py
        │   ├── model_fixtures.py
        │   └── ...
        └── data/


``fixtures/`` contains all the data that is used for tests comparing real and
expected output of functions, as well as the files that creates the data. It is
split into ``unit/`` and ``integration/`` test fixtures. These subdirectories
contain the scripts that produce the respective fixtures using the parameters
defined in ``config/`` and store them in ``data/``. The ``Snakefile`` defines a
workflow to create all fixtures at once using the ``config.yaml``. See
`Fixture creation workflow`_ for more details on that.

Unit tests
**********

.. code::

    unit/
    ├── general/
    │   ├── test_input_output.py
    │   ├── test_network_properties.py
    │   └── test_utils.py
    ├── lif/
    │   ├── test_delta.py
    │   ├── test_exp.py
    │   └── test_lif.py
    └── models/
        ├── test_microcircuit.py
        └── test_network.py


``unit/`` contains all unit tests. It is split according to the submodules of
``nnmt``.

Integration tests
*****************

.. code::

    integration/
    ├── test_functionality.py
    ├── test_internal_storage.py
    ├── test_reproduce_Bos2016.py
    ├── test_reproduce_Schuecker2015.py
    └── test_usage_examples.py


``integration/`` contains all integration tests. They are seperated by types of
integration tests.

Meta tests
**********

.. code::

    meta/
    └── test_checks.py

``meta/`` contains tests for custom assert functions.


Test Design
===========

Many test classes define the tested function as ``staticmethod``, because the
function itself is not tightly related to class, but we still want to attach it
to the class for later reference. This allows us to call the function as an
'unbound function', without passing the instance to the function:
``self.func()`` = ``func()`` != ``func(self)``.

There are two special fixtures that are definded in ``conftest.py``:

If a test requires the ``pos_keys`` fixture, it will be parametrized such that
it tests all positive arguments the tested function (defined as a
``staticmethod`` of the test class) takes. The list of all possible positive
arguments is defined within ``conftest.py``.

If a test requires ``output_test_fixtures``, pytest will pass the output
fixtures corresponding to the ``output_key`` defined as a test class variable.
Those output key results are checked into the repository for convenience, but
can be created from the sources (see Fixture Creation Workflow). This allows us
to parametrize the tests such that the function is tested in different
parameter regimes (e.g. mean-driven regime vs. fluctuation-driven regime).

Fixture Creation Workflow
=========================

Fixture creation is a sensible part of the testing framework as it supplies a
kind of ground truth to test against. Please make sure that your code is
trustworthy before running the fixture creation. Otherwise, tests might
incorrectly fail or pass.

The fixture creation workflow is defined using
`Snakemake <https://snakemake.readthedocs.io/en/stable/index.html>`_, a
workflow management system using a Python based syntax. It is recommended to
install it in a separate conda environment (see
`Installation <https://snakemake.readthedocs.io/en/stable/getting_started/installation.html>`_).

To invoke the workflow and create the fixtures using the same conda environment
you are using the toolbox with, you first need to export the conda environment.
Therefore set ``tests/fixtures/envs`` as current working directory, activate the
corresponding conda environment and type

.. code::

    conda env export -f environment.yaml

Then open the created ``environment.yaml`` file, remove the last line starting
with `prefix` and add the line ``- -e ../../../../`` to the list at the end.
Change your current working directory to ``tests/fixtures``, activate the conda
environment you have installed sakemake in and type

.. code::

    snakemake --use-conda --cores 1

The workflow then takes care of installing the necessary conda environments and
creating all fixtures that are specified within ``tests/fixtures/config.yaml``.
By default, the workflow looks whether the requested fixtures exists and only
creates them if they don't.

It might be useful to first see what the workflow is planning to do by
triggering a 'dry-run' with: ``snakemake -n``. Furthermore the execution of
single rules can be enforced with the ``-R`` flag, e.g.:

.. code::

    snakemake --use-conda --cores 1 -R make_Bos2016_data

This is useful if one specific fixture should be re-created.

Have a look at the `Snakemake Documentation
<https://snakemake.readthedocs.io/en/stable/index.html>`_ for more information.