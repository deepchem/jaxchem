# flake8: noqa
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
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


import sphinx_rtd_theme
import jaxchem


# -- Project information -----------------------------------------------------

project = 'JAXChem'
copyright = '2020, deepchem-contributors'
author = 'deepchem-contributors'

# The full version, including alpha/beta/rc tags
version = jaxchem.__version__
release = jaxchem.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.napoleon',
    'sphinx.ext.linkcode', 'sphinx.ext.mathjax',
]

autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': True,
    'exclude-members': '__repr__, __str__, __weakref__',
}

autodoc_typehints = "description"

mathjax_path = 'http://mathjax.connectmv.com/MathJax.js?config=default'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
}

html_context = {
    "display_github": True,
    "github_user": "deepchem",
    "github_repo": "jaxchem",
    "github_version": "master",
    "conf_py_path": "/docs/source/",
}


# Thanks to https://github.com/materialsproject/pymatgen/blob/master/docs_rst/conf.py
def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info['module']]
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)
        import inspect
        import os
        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != 'py' or not info['module']:
        return None

    try:
        filename = 'jaxchem/%s#L%d-L%d' % find_source()
    except:
        filename = info['module'].replace('.', '/') + '.py'

    tag = 'v' + jaxchem.__version__
    return "https://github.com/deepchem/jaxchem/blob/%s/%s" % (tag, filename)


def setup(app):
    def skip(app, what, name, obj, skip, options):
        members = [
            '__init__',
            '__call__',
        ]
        return False if name in members else skip

    app.connect('autodoc-skip-member', skip)
