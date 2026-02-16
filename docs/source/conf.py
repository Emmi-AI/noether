#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import os
import sys
from datetime import datetime

# Path to src/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

project = "Noether Framework"
author = "Emmi AI"
copyright = f"{datetime.now():%Y}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    # "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "autoapi.extension",  # <-- use AutoAPI
    "sphinx_copybutton",  # for "copy" buttons on code blocks
    # Optional if you want Markdown pages:
    # Optional if you want CLI docs:
    # "sphinx_click",
]
extensions += ["sphinx_design"]
extensions += ["myst_parser"]  # for Markdown code blocks parsing
# --- THEME CUSTOMIZATION:
html_theme = "furo"

EMMI_THEME = {
    # Light — PURPLE
    "light_css_variables": {
        # Brand + links
        "color-brand-primary": "#661371",
        "color-brand-content": "#661371",
        "color-link": "#4B0357",
        "color-link--hover": "#BC48BF",
        "color-link--visited": "#DB8AFF",  # washed-out (lighter)
        # Text + surfaces
        "color-foreground-primary": "#222222",
        "color-foreground-secondary": "#555555",
        "color-background-primary": "#FFFFFF",
        "color-background-secondary": "#F7F7F9",
        "color-sidebar-background": "#FAFAFC",
        # Code
        "color-code-background": "#F5F7F9",
        "color-code-foreground": "#222222",
        # Sidebar hierarchy + states
        "color-sidebar-link-text--top-level": "#661371",
        "color-sidebar-link-text": "#555555",  # 2nd level
        "color-sidebar-link-text--hover": "#BC48BF",
        "color-sidebar-link-text--current": "#661371",
        "color-sidebar-link-background--hover": "#EEEAF1",
        "color-background-border": "rgba(0,0,0,0.08)",
    },
    # Dark — TEAL
    # "dark_css_variables": {
    #     # Brand + links
    #     "color-brand-primary": "#00A392",
    #     "color-brand-content": "#00A392",
    #     "color-link": "#00A392",
    #     "color-link--hover": "#00EAD3",
    #     "color-link--visited": "#007367",  # washed-out (darker)
    #     # Text + dark greys
    #     "color-foreground-primary": "#EDEDED",
    #     "color-foreground-secondary": "#BEBEBE",
    #     "color-background-primary": "#1E1E1E",
    #     "color-background-secondary": "#2A2A2A",
    #     "color-sidebar-background": "#242424",
    #     # Code
    #     "color-code-background": "#1B1B1B",
    #     "color-code-foreground": "#EDEDED",
    #     # Sidebar hierarchy + states
    #     "color-sidebar-link-text--top-level": "#00A392",
    #     "color-sidebar-link-text": "#CFCFCF",  # 2nd level
    #     "color-sidebar-link-text--hover": "#00EAD3",
    #     "color-sidebar-link-text--current": "#00A392",
    #     "color-sidebar-link-background--hover": "#a1a1a1",
    #     # "color-sidebar-item-background": "",
    #     "color-background-border": "rgba(255,255,255,0.08)",
    # },
    "dark_css_variables": {
        # Brand + links
        "color-brand-primary": "#661371",
        "color-brand-content": "#661371",
        "color-link": "#4B0357",
        "color-link--hover": "#BC48BF",
        "color-link--visited": "#DB8AFF",  # washed-out (lighter)
        # Text + surfaces
        "color-foreground-primary": "#222222",
        "color-foreground-secondary": "#555555",
        "color-background-primary": "#FFFFFF",
        "color-background-secondary": "#F7F7F9",
        "color-sidebar-background": "#FAFAFC",
        # Code
        "color-code-background": "#F5F7F9",
        "color-code-foreground": "#222222",
        # Sidebar hierarchy + states
        "color-sidebar-link-text--top-level": "#661371",
        "color-sidebar-link-text": "#555555",  # 2nd level
        "color-sidebar-link-text--hover": "#BC48BF",
        "color-sidebar-link-text--current": "#661371",
        "color-sidebar-link-background--hover": "#EEEAF1",
        "color-background-border": "rgba(0,0,0,0.08)",
    },
}
with open(os.path.join("_static", "Emmi_AI_logomark_black.svg")) as f:
    emmi_logo = f.read()

html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    **EMMI_THEME,
    # "logo": {},  # create an empty dict for potential light/dark modes updates
    "source_repository": "https://github.com/Emmi-AI/noether",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "footer_icons": [
        {"name": "Emmi AI", "url": "https://www.emmi.ai/", "html": emmi_logo},
        {
            "name": "GitHub",
            "url": "https://github.com/Emmi-AI/noether",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# Syntax highlighting that pairs well with both palettes
pygments_style = "friendly"  # light
pygments_dark_style = "friendly"  # dark
# pygments_dark_style = "monokai"  # dark

# ---

# Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_attr_annotations = True

# Types in param blocks
autodoc_typehints = "description"
autodoc_typehints_format = "short"
typehints_fully_qualified = False
always_document_param_types = True

# Let AutoAPI produce API pages from src/
autoapi_type = "python"
autoapi_dirs = [SRC]
autoapi_add_toctree_entry = False
autoapi_member_order = "bysource"  # preserve source order
autoapi_python_class_content = "both"  # class doc + __init__ doc together
autoapi_keep_files = False
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",  # this adds re-exports in __init__.py, giving a cleaner API structure
    "no-private-members",
    "no-special-members",
    "no-module-attributes",
]
autoapi_ignore = [
    "**/tests/**",
    "**/docs/**",
    "**/*.ipynb",
    "**/*.md",
    "**/.venv/**",
]


# Do NOT have autosummary import modules
autosummary_generate = False

# Intersphinx (drop typer to avoid 404)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "boto3": ("https://boto3.amazonaws.com/v1/documentation/api/latest/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # If old apidoc output exists, ignore it:
    "emmi_inference/generated/**",
    "noether/generated/**",
    "**/*.ipynb",
    "**/*.md",
]
html_static_path = ["_static"]
html_css_files = ["emmi_docs_theme.css"]
html_js_files = ["force_light_theme.js"]
html_logo = "_static/Emmi_AI_logo_black.svg"  # this works
# This doesn't work, not sure how to make it light/dark mode agnostic yet:
# if isinstance(html_theme_options["logo"], dict):
#     html_theme_options["logo"].update({
#         "image_light": "_static/Emmi_AI_logo_black.svg",
#         "image_dark": "_static/Emmi_AI_logo_green.svg",
#     })


def skip_handler(app, what, name, obj, skip, options):
    if what == "class":
        # Check if the docstring exists and matches the parent
        if (
            obj.docstring
            and type(obj.docstring) == str
            and obj.docstring.startswith('!!! abstract "Usage Documentation"')
        ):
            obj.docstring = ""  # Wipe the inherited docstring

    if getattr(obj, "inherited", False):
        # Skip all inherited attributes from BaseModel (pydantic) to reduce noise
        if "BaseModel" in name or "pydantic" in name:
            return True

    WHITELIST = [
        "_sampler_config_from_key",
    ]

    # Allow certain private methods that are important for understanding the class behavior
    if what == "method" and name.split(".")[-1] in WHITELIST and not getattr(obj, "inherited", False):
        return False
    return skip


def setup(app):
    app.connect("autoapi-skip-member", skip_handler)
