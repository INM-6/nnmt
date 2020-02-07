from setuptools import setup

setup(name = 'lif_meanfield_tools',
      version = '0.1',
      description = 'Mean-field toolbox for networks of LIF neurons.',
      long_description = open('README.md').read(),
      url = 'https://github.com/INM-6/lif_meanfield_tools',
      author = 'see authors.md',
      author_email = 'm.layer@fz-juelich.de',
      license = 'GNU GPLv3',
      packages = ['lif_meanfield_tools'],
      install_requires = [
        'setuptools>=23.1.0',
        'numpy>=1.8',
        'scipy>=0.14',
        'Cython>=0.20',
        'h5py>=2.5',
        'pint',
        'h5py_wrapper',
        'pyyaml',
        'requests',
        'mpmath',
        'decorator'],
     python_requires='>=3')
