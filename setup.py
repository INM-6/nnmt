from setuptools import setup, find_packages

setup(name='nnmt',
      version='1.1.0',
      description='Neuronal Network Meanfield Toolbox',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      url='https://github.com/INM-6/nnmt',
      author='see docs/source/authors_and_contributors.rst',
      author_email='m.layer@fz-juelich.de',
      license='GNU GPLv3',
      packages=find_packages(include=['nnmt', 'nnmt.*']),
      install_requires=[
        'setuptools>=23.1.0',
        'numpy>=1.8',
        'scipy>=0.14',
        'Cython>=0.20',
        'h5py>=2.5',
        'matplotlib>=2.0',
        'pint',
        'pyyaml',
        'requests',
        'mpmath',
        'decorator',
        'pytest>=5.4',
        'pytest-mock>=3.1'],
      python_requires='>=3')
