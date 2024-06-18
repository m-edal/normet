# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = 'NORmet'
author = 'Dr. Congbo Song and other MEDAL group members'
release = 'v0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'recommonmark',
    'sphinx_markdown_tables',
    'sphinx.ext.githubpages',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting',
    'numpydoc',
]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'


# -- Options for EPUB output
epub_show_urls = 'footnote'
