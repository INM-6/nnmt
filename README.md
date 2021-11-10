# NNMT - Neuronal Network Meanfield Toolbox

NNMT is an open-source, community centered Python package for collecting
reusable implementations of analytical methods for neuronal network model
analysis based on mean-field theory.

#### Documentation

Please visit our [official documentation](<link to official doc>).

In order to compile the documentation on your own, you have to change your
working directory to ``nnmt/`` and install and activate the provided conda
environment

.. code:: bash

  conda env create -f environment.yaml
  conda activate nnmt

Change you working directory to ``nnmt/docs/`` and run the following commands

.. code:: bash

  make clean
  make html

This will compile the documentation and create the folder ``build/``.
Now you can access the documentation using your preferred browser by opening
the file ``build/html/index.html``.

#### Citation

If you use NNMT for your project, please don't forget to
[cite NNMT](docs/source/citing.rst).

#### License

GNU General Public License v3.0, see [LICENSE](docs/source/license.rst) for
details.

#### Acknowledgments

See [acknowledgments](docs/source/acknowledgments.rst).
