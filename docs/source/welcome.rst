This Python package provides useful tools for analyzing neuronal networks
consisting of leaky integrate-and-fire (LIF) neurons. These tools are based on
mean-field theory of neuronal networks. That is why this package is called
nnmt (NNMT).

The package provides implementations used in the same or a similar version in
the following scientific publications:

.. - `Fourcaud & Brunel (2002) <https://doi.org/10.1162/089976602320264015>`_
.. - `Schuecker et al. (2014) <https://arxiv.org/abs/1410.8799>`_
.. - `Schuecker et al. (2015) <https://doi.org/10.1103/PhysRevE.92.052119>`_
.. - `Schuecker et al. (2017) <https://doi.org/10.1371/journal.pcbi.1005179>`_
.. - `Bos et al. (2016) <https://dx.doi.org/10.1371%2Fjournal.pcbi.1005132>`_
- :cite:t:`fourcaud2002`
- :cite:t:`schuecker2014`
- :cite:t:`schuecker2015`
- :cite:t:`bos2016`
- :cite:t:`senk2020`

Using this package, you can easily calculate quantities like firing rates, power
spectra, and many more, which give you a deeper and more intuitive understanding
of what your network does. If your network is not behaving the way you want it
to, these tools might help you to figure out, or even tell you, what you need to
change in order to achieve the desired behaviour. It is easy to store
(and in the future, to plot) results and reuse them for further analyses.

The package is alive. We are continuously trying to improve and simplify its
usage.
We are always happy about feedback. So please do not hesitate to contact us.
If you encounter a problem or have a feature request, you can open an
`Issue <https://github.com/INM-6/nnmt/issues>`_.
Contributions are always welcome via
`Pull requests <https://github.com/INM-6/nnmt/pulls>`_.

If you are using this toolbox, please cite us: for a specific release, we
recommend to use the reference from `Zenodo <https://zenodo.org/>`_. Otherwise,
you can also provide a link to this repository with the hash of the respective
commit. In addition, please also cite the publications that used the methods
implemented here first. In [How to Use This Package](#how-to-use-this-package)
you can find details on which function of this package refers to which
publication.

.. image:: readme_figures/power_spectra.png
  :width: 600
  :alt: picture of power spectra

The figure shows power spectra calculated with this toolbox using the minimal
example script ``examples/power_spectra.py``.
