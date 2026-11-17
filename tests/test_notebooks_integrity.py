"""Integrity and syntax validation tests across all notebooks in the repository."""

import ast
import glob
import json
import os
import pytest


def get_all_notebook_paths():
    """Discover all notebook files in repository."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pattern = os.path.join(root_dir, "**", "*.ipynb")
    paths = sorted(glob.glob(pattern, recursive=True))
    assert len(paths) >= 5, f"Expected at least 5 notebooks, found {len(paths)}"
    return paths


@pytest.mark.parametrize("nb_path", get_all_notebook_paths())
def test_notebook_json_validity(nb_path: str):
    """Ensure every notebook is a valid JSON document conforming to nbformat schema."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb, f"Notebook {nb_path} missing 'cells' key"
    assert "metadata" in nb, f"Notebook {nb_path} missing 'metadata' key"
    assert "nbformat" in nb, f"Notebook {nb_path} missing 'nbformat' key"
    assert isinstance(nb["cells"], list), f"'cells' must be a list in {nb_path}"


@pytest.mark.parametrize("nb_path", get_all_notebook_paths())
def test_notebook_code_cells_syntax(nb_path: str):
    """Ensure all code cells in every notebook compile into valid Python AST with zero syntax errors."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell_idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            clean_lines = [
                line
                for line in src.splitlines()
                if not line.strip().startswith("!")
                and not line.strip().startswith("%")
            ]
            code_str = "\n".join(clean_lines)
            try:
                ast.parse(code_str)
            except SyntaxError as e:
                pytest.fail(
                    f"Syntax error in {os.path.basename(nb_path)} cell {cell_idx} at line {e.lineno}: {e.msg}"
                )
