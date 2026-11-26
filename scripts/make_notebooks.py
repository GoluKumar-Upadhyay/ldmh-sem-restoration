"""Builds the .ipynb files from the cell definitions (keeps the notebooks clean and regenerable).
Usage:  python scripts/make_notebooks.py"""
import os
import sys
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from defs_a import NB as A
from defs_b import NB as B
from defs_c import NB as Cc

OUT = os.path.join(os.path.dirname(HERE), "notebooks")
os.makedirs(OUT, exist_ok=True)
for name, cells in {**A, **B, **Cc}.items():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
    nbf.write(nb, os.path.join(OUT, name))
    print("wrote", name, f"({len(cells)} cells)")
