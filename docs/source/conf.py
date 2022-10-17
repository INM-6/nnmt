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

project = 'NNMT'
copyright = '2022, Moritz Layer, Johanna Senk, Simon Essink, Alexander van Meegen, Hannah Bos, Moritz Helias'
author = 'Moritz Layer, Johanna Senk, Simon Essink, Alexander van Meegen, Hannah Bos, Moritz Helias'

# The full version, including alpha/beta/rc tags
release = '1.2.0'


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
    'sphinx_gallery.gen_gallery',
    # 'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex',
    'numpydoc',
    'sphinx_tabs.tabs',
    ]
#
# autoapi_dirs = ['../../nnmt']
# autoapi_type = 'python'
# autoapi_options  = [
#     'members',
#     'undoc-members']

#
# napoleon_google_docstring = False
# napoleon_numpy_docstring = True
# napoleon_use_admonition_for_notes = True
# napoleon_use_param = False

sphinx_gallery_conf = {
     'examples_dirs': '../../examples',  # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
     'plot_gallery': 'False'
}

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


# path to bibfiles
bibtex_bibfiles = ['bib/nnmt.bib']
# bibtex style
bibtex_default_style = 'plain'
# inline reference style
bibtex_reference_style = 'author_year'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'
#
html_theme_options = {
    'font_family': 'Arial',
    'page_width': '1120px',  # default is 940
    'description': 'Neuronal Network Meanfield Toolbox',
    'body_max_width': '900px',
    'sidebar_width': '220px',  # default is 220
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# The name of the image file (relative to this directory) to place at the top
# of the sidebar
# html_logo = 'images/logo.png'

# html_copy_source = False

# html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True


# Suppresses  wrong numpy doc warnings
# see here https://github.com/phn/pytpm/issues/3#issuecomment-12133978
numpydoc_show_class_members = False

# A fix for Alabaster theme for no space between a citation reference
# and citation text
# https://github.com/sphinx-doc/sphinx/issues/6705#issuecomment-536197438
html4_writer = True

#
# def setup(app):
#     app.add_css_file('my_theme.css')
