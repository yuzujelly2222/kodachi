"""Sphinx configuration for kodachi."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import kodachi  # noqa: E402

project = "kodachi"
author = "yuzujelly2222"
copyright = "2026, yuzujelly2222"
release = kodachi.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "myst_parser",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_default_options = {"members": True, "special-members": "__init__"}

napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "furo"
html_title = f"kodachi {release}"
html_theme_options = {
    "source_repository": "https://github.com/yuzujelly2222/kodachi/",
    "source_branch": "main",
    "source_directory": "docs_src/",
}
