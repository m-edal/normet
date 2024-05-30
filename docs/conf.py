# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = 'NORmet'
author = 'Dr. Congbo Song and other MEDAL group members'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'
html_static_path = ['_static']
