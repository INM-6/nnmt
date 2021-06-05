# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# import os
# import sys
# sys.path.insert(0, os.path.abspath('../..'))  # Source code dir relative to this file


# -- Project information -----------------------------------------------------

project = 'lif_meanfield_tools'
copyright = '2021, Moritz Layer, Johanna Senk, Simon Essink, Alexander van Meegen, Karolína Korvasová, Hannah Bos, Jannis Schuecker, Moritz Helias'
author = 'Moritz Layer, Johanna Senk, Simon Essink, Alexander van Meegen, Karolína Korvasová, Hannah Bos, Jannis Schuecker, Moritz Helias'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.imgmath',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    # 'sphinx.ext.napoleon',
    'numpydoc',
    'sphinx_tabs.tabs',
    ]
#
# autoapi_dirs = ['../../lif_meanfield_tools']
# autoapi_type = 'python'
# autoapi_options  = [
#     'members',
#     'undoc-members']

#
# napoleon_google_docstring = False
# napoleon_numpy_docstring = True
# napoleon_use_admonition_for_notes = True
# napoleon_use_param = False


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

#
# autodoc_default_options = {
#     'members': True,
#     'undoc-members': True,
#     'inherited-members': True
# }


# Required to automatically create a summary page for each function listed in
# the autosummary fields of each module.
autosummary_generate = True

# Set to False to not overwrite the custom _toctree/*.rst
autosummary_generate_overwrite = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
