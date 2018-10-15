from setuptools import setup

long_description = open("README.md").read()

setup(name='lif_meanfield_tools',
      version='0.1',
      description='Mean-field toolbox for networks of LIF neurons.',
      url='https://github.com/INM-6/lif_meanfield_tools',
      author='Moritz Layer, Hannah Bos, Jannis Schuecker, Johanna Senk, Karolina Korvasova, Moritz Helias',
      author_email='m.layer@fz-juelich.de',
      license='LICENSE',
      install_requires = [ # TODO
        'setuptools>=23.1.0',
        'numpy>=1.8',
        'scipy>=0.14',
        'Cython>=0.20',
        'h5py>=2.5',
])

