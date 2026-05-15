# Configuration file for the Sphinx documentation builder.

import os
import sys

repo_root = os.path.abspath('..')
sys.path.insert(0, repo_root)

# -- Project information -----------------------------------------------------

project = 'bayRing'
copyright = '2023 onwards, Gregorio Carullo, Marina De Amicis, Jaime Redondo Yuste'
author = 'Gregorio Carullo, Marina De Amicis, Jaime Redondo-Yuste'

try:
    import bayRing
    version = bayRing.__version__
    release = bayRing.__version__
except Exception:
    version = '1.0.0'
    release = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

try:
    import myst_parser  # noqa: F401
    extensions.append('myst_parser')
except Exception:
    pass

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = ['.rst', '.md']

# -- Options for HTML output -------------------------------------------------

pygments_style = 'friendly'

try:
    import sphinx_rtd_theme  # noqa: F401
    html_theme = 'sphinx_rtd_theme'
except Exception:
    html_theme = 'alabaster'

html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 2,
    'sticky_navigation': True,
    'titles_only': True,
}

html_title = 'bayRing documentation'
html_logo = '_static/bayRing_docs_image.svg'
htmlhelp_basename = 'bayRingdocs'
html_static_path = ['_static']
html_css_files = ['css/custom.css']

mathjax_path = (
    'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js'
    '?config=TeX-AMS_SVG'
)
mathjax_config = {
    'messageStyle': 'none',
    'SVG': {
        'font': 'STIX-Web',
        'scale': 96,
        'linebreaks': {'automatic': True},
    },
    'TeX': {
        'equationNumbers': {'autoNumber': 'none'},
    },
}
