#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""Sphinx extension that adds a ``:source:`` role for viewing file contents.

Usage (same syntax as ``:download:``):

    :source:`display text <relative/path/to/file>`
    :source:`relative/path/to/file`

Instead of downloading the file, clicking the link opens a generated page
that displays the file contents with syntax highlighting and line numbers.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from docutils import nodes
from sphinx import addnodes
from sphinx.util.docutils import SphinxRole

GENERATED_DIR = "_generated/source_files"

LANG_MAP = {
    ".py": "python",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".cfg": "ini",
    ".ini": "ini",
    ".sh": "bash",
    ".bash": "bash",
    ".rst": "rst",
    ".md": "markdown",
}


def _get_language(filepath: str) -> str:
    return LANG_MAP.get(Path(filepath).suffix.lower(), "text")


def _page_name(abs_path: Path, project_root: Path) -> str:
    """Create a unique, filesystem-safe page name from a resolved file path."""
    try:
        rel = abs_path.relative_to(project_root)
    except ValueError:
        rel = Path(abs_path.name)
    return str(rel).replace(os.sep, "__").replace("/", "__").replace(".", "_")


def _scan_and_generate(app):
    """Pre-scan RST files for :source: roles and generate view pages."""
    srcdir = Path(app.srcdir)
    project_root = srcdir.parent.parent  # docs/source/ -> project root
    gen_dir = srcdir / GENERATED_DIR

    # Clean up previous generated files to avoid stale pages
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Match :source:`title <path>` or :source:`path`
    pattern = re.compile(r":source:`(?:[^<`]*<([^>]+)>|([^`]+))`")

    generated: set[str] = set()

    for rst_file in srcdir.rglob("*.rst"):
        text = rst_file.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            rel_path = (match.group(1) or match.group(2)).strip()
            abs_path = (rst_file.parent / rel_path).resolve()

            if not abs_path.is_file():
                continue

            name = _page_name(abs_path, project_root)
            if name in generated:
                continue
            generated.add(name)

            gen_file = gen_dir / f"{name}.rst"
            literalinclude_path = os.path.relpath(abs_path, gen_file.parent)
            filename = abs_path.name
            lang = _get_language(str(abs_path))
            title_underline = "=" * len(filename)

            gen_file.write_text(
                f"{filename}\n"
                f"{title_underline}\n"
                "\n"
                f"``{abs_path.relative_to(project_root)}``\n"
                "\n"
                f".. literalinclude:: {literalinclude_path}\n"
                f"   :language: {lang}\n"
                "   :linenos:\n",
                encoding="utf-8",
            )

    # Keep generated files out of version control
    (gen_dir / ".gitignore").write_text("*\n")


class SourceViewRole(SphinxRole):
    """A role that links to an auto-generated page showing file contents."""

    def run(self):
        text = self.text.strip()
        m = re.match(r"^(.+?)\s*<(.+?)>\s*$", text)
        if m:
            title, rel_path = m.group(1).strip(), m.group(2).strip()
        else:
            title = os.path.basename(text)
            rel_path = text

        # Resolve the referenced file to an absolute path
        docdir = Path(self.env.doc2path(self.env.docname)).parent
        abs_path = (docdir / rel_path).resolve()
        srcdir = Path(self.env.srcdir)
        project_root = srcdir.parent.parent

        name = _page_name(abs_path, project_root)
        target_docname = f"{GENERATED_DIR}/{name}"

        refnode = addnodes.pending_xref(
            "",
            nodes.inline("", title),
            refdomain="std",
            reftype="doc",
            reftarget="/" + target_docname,
            refexplicit=True,
        )
        return [refnode], []


def setup(app):
    app.add_role("source", SourceViewRole())
    app.connect("builder-inited", _scan_and_generate)
    return {"version": "0.1", "parallel_read_safe": True}
