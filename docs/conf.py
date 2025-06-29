"""Sphinx configuration for the VXDF documentation site.

To build the docs locally:

    pip install -r requirements.txt  # ensure sphinx and theme are available
    cd docs && make html

The generated HTML will be placed in ``docs/_build/html``.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup – add the project root to ``sys.path`` so ``autodoc`` can import
# the ``vxdf`` package without an editable install.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = "VXDF"
author = "VXDF Developers"
copyright = f"{datetime.now().year}, {author}"

# The full version, including alpha/beta/rc tags
try:
    # Lazy import to avoid importing heavy deps when building docs on RTD
    from vxdf import __version__ as release  # type: ignore
except Exception:
    release = "0.0.0"

# ---------------------------------------------------------------------------
# General Sphinx settings
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # parse NumPy/Google style docstrings
    "sphinx.ext.intersphinx",
]

autosummary_generate = True  # create stubs automatically

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
}

# List of patterns to ignore when looking for source files.
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------
# HTML output options
# ---------------------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
