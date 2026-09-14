# Sphinx configuration for CVineMarketGen (built by Read the Docs).
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'CVineMarketGen'
author = 'Mamadou Yamar Thioub'
copyright = '2026, Mamadou Yamar Thioub'
release = '0.2.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_nb',
]

# The notebooks live in ../examples; copy them next to the docs sources at
# build time so myst-nb can render them, with their stored outputs (no execution).
_here = os.path.dirname(__file__)
_src = os.path.join(_here, '..', 'examples')
_dst = os.path.join(_here, 'examples')
os.makedirs(_dst, exist_ok=True)
for _f in os.listdir(_src):
    if _f.endswith('.ipynb'):
        shutil.copy(os.path.join(_src, _f), os.path.join(_dst, _f))
nb_execution_mode = 'off'
myst_enable_extensions = ['dollarmath', 'colon_fence']

autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_default_options = {'members': True, 'undoc-members': False, 'show-inheritance': True}
napoleon_numpy_docstring = True
napoleon_google_docstring = False
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'superpowers']

html_theme = 'sphinx_rtd_theme'
html_title = 'CVineMarketGen'
html_static_path = ['_static']
html_copy_source = False
html_show_copyright = False
html_show_sphinx = False
add_module_names = False
pygments_style = 'sphinx'
html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': False,
    'sticky_navigation': True,
}
html_context = {
    'display_github': True,
    'github_user': 'mamadouyamar',
    'github_repo': 'CVineMarketGen',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}
