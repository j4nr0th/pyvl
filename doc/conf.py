"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html"""

import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyVL'
copyright = '2024, Jan Roth'
author = 'Jan Roth'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions: list[str] = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "jupyter_sphinx",
    "pydata_sphinx_theme",
    "hawkmoth",
    "hawkmoth.ext.javadoc",
]

hawkmoth_root = os.path.abspath("../src/core")


def _hawkmoth_clang_flags():
    """Build hawkmoth clang flags with system include paths queried from clang."""
    flags = ["-I.", "-DCVL_ARRAY_ARG(arr,sz)=*arr"]
    try:
        import subprocess
        result = subprocess.run(
            ["clang", "-E", "-x", "c", "-", "-v"],
            capture_output=True, text=True, input="", timeout=10,
        )
        # Parse the search path list from stderr
        lines = result.stderr.splitlines()
        in_search = False
        for line in lines:
            if line.startswith("#include <...> search starts here:"):
                in_search = True
                continue
            if in_search:
                path = line.strip()
                if not path or "End of search list" in path:
                    break
                flags.append(f"-I{path}")
    except Exception:
        pass
    return flags


hawkmoth_clang = _hawkmoth_clang_flags()
hawkmoth_transform_default = "javadoc"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# -- Options for Intersphinx -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pyvista": ("https://docs.pyvista.org/" , None),
}

# -- Options for Napoleon ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for Autodoc -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html


autodoc_member_order = "groupwise"
autodoc_type_aliases = {
    "npt.ArrayLike": "array_like",
    "npt.NDArray": "ndarray",
    "VecLike3" : "Vec3",
}

# -- Options for Sphinx Gallery ----------------------------------------------
# https://sphinx-gallery.github.io/stable/index.html
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "reference_url": {
         # The module you locally document uses None
        "pyvl": None,
    },
    "image_scrapers": ("matplotlib", "pyvista"),
}
import pyvista
pyvista.BUILDING_GALLERY = True
