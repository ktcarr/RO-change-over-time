"""
Tests for src/convert_marimo.py: converting a notebook must keep every
comment where it was, and must otherwise give the same code as stock
`marimo convert`.

Run from the repository root:
    envs/bin/python -m pytest tests/test_convert_marimo.py
"""

import ast
import json
import subprocess
import sys

from marimo._convert.converters import MarimoConvert

import src.convert_marimo

## cells covering each kind of rewrite marimo makes, all with comments:
## - `fig` defined in several cells, each using only its own (made
##   cell-local: `_fig`), once after non-ASCII text on the same line (AST
##   columns count bytes, not characters)
## - `total` redefined in a later cell that is read afterwards (numbered:
##   `total_1`), via `+=` with a multi-line, commented value (expanded to
##   `total_1 = total + ...` first)
## - an import repeated in a later cell (removed there, comment kept)
## - a function redefined in a later cell (numbered: `helper_1`)
CELLS = [
    "## imports\nimport numpy as np  # arrays\n",
    "## first figure\nfig = 1  # placeholder\nprint(fig)\n",
    "## second figure\nfig = 2\n## show it\nprint(fig)\n",
    '## label = "Niño 3.4"\nlabel = "Niño"; fig = 3  # after non-ASCII text\nprint(fig)\n',
    "## a running total\ntotal = np.zeros(3)\n",
    "## add to it (multi-line value)\ntotal += (\n    1  # one\n    + 2\n)\n",
    "## read the new total\nprint(total)\n",
    "## imported again\nimport numpy as np  # repeated\nx = np.ones(2)\n",
    "## a helper\ndef helper():\n    return 1\n",
    "## redefined helper\ndef helper():\n    ## two now\n    return 2\n",
    "## use the helper\nprint(helper(), label, x)\n",
]

## lines the conversion must produce (indented one level inside the cell
## function); each keeps its comment on the same code as in the notebook
EXPECTED_LINES = [
    "    _fig = 1  # placeholder",
    "    ## show it\n    print(_fig)",
    '    label = "Niño"; _fig = 3  # after non-ASCII text',
    "    total_1 = total + (\n        1  # one\n        + 2\n    )",
    "    ## imported again\n    # repeated\n    x = np.ones(2)",
    "def helper_1():\n    ## two now\n    return 2",
]


def make_notebook(cells):
    """
    A minimal .ipynb (as JSON text) with one code cell per source string.

    Args:
        - cells (list of str): code cell sources

    Returns:
        - text (str): notebook JSON
    """
    notebook = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    ## one code cell per source, without outputs
    for source in cells:
        cell = {
            "cell_type": "code",
            "metadata": {},
            "source": source,
            "outputs": [],
            "execution_count": None,
        }
        notebook["cells"].append(cell)
    ## serialized, as read from an .ipynb file
    text = json.dumps(notebook)
    return text


def test_keeps_every_comment():
    """Every comment of the notebook's code cells is in the output, in order."""
    notebook = make_notebook(CELLS)
    marimo_text = src.convert_marimo.convert(notebook)
    ## comments not found in order in the output (should be none)
    missing = src.convert_marimo.check_comments(notebook, marimo_text)
    assert missing == []


def test_comments_stay_in_place():
    """Comments stay on (or above) the same code, with the renames made."""
    marimo_text = src.convert_marimo.convert(make_notebook(CELLS))
    for expected in EXPECTED_LINES:
        assert expected in marimo_text, expected


def test_same_code_as_marimo():
    """
    Apart from comments and formatting, the output is the same code as
    stock `marimo convert`: identical syntax trees.
    """
    notebook = make_notebook(CELLS)
    ours = src.convert_marimo.convert(notebook)
    stock = MarimoConvert.from_ipynb(notebook).to_py()
    ## ast.dump ignores comments and formatting, and compares everything else
    assert ast.dump(ast.parse(ours)) == ast.dump(ast.parse(stock))


def test_runs_as_marimo_app(tmp_path):
    """
    The converted notebook runs top to bottom as a script (marimo accepts
    every cell, and the renamed variables connect the cells correctly).

    Args:
        - tmp_path (pathlib.Path): pytest's temporary directory
    """
    ## marimo reads cell code from the file, so write one
    fp = tmp_path / "converted.py"
    fp.write_text(src.convert_marimo.convert(make_notebook(CELLS)))

    ## run it with this Python; marimo exits non-zero if a cell fails
    result = subprocess.run([sys.executable, fp], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    ## the last cell prints helper_1(), the label and x = np.ones(2)
    assert "2 Niño [1. 1.]" in result.stdout


## marimo cells for end_figure_cells: (cell body, expected body)
FIGURE_CELLS = [
    ## a final plt.show() becomes plt.gca(), keeping its comment
    (
        "    fig = plt.figure()\n    plt.show()  # display\n",
        "    fig = plt.figure()\n    plt.gca()  # display\n",
    ),
    ## a figure never shown gets plt.gca() before the return
    (
        "    fig, ax = plt.subplots()\n    ax.plot([1, 2])\n",
        "    fig, ax = plt.subplots()\n    ax.plot([1, 2])\n    plt.gca()\n",
    ),
    ## several figures (plt.show() mid-cell): left for splitting by hand
    (
        "    for i in range(2):\n        plt.figure()\n        plt.show()\n    x = 1\n",
        "    for i in range(2):\n        plt.figure()\n        plt.show()\n    x = 1\n",
    ),
    ## a helper that makes figures, but the cell itself draws nothing
    (
        "    def make():\n        return plt.subplots()\n",
        "    def make():\n        return plt.subplots()\n",
    ),
]


def test_end_figure_cells():
    """Figure cells end with plt.gca(); other cells are unchanged."""
    for body, expected in FIGURE_CELLS:
        ## one hidden cell (decorator with arguments) around each body
        header = "import marimo\napp = marimo.App()\n\n@app.cell(hide_code=True)\n"
        cell = header + "def _(plt):\n" + body + "    return\n"
        result = src.convert_marimo.end_figure_cells(cell)
        ## only the cell body may change
        assert result == header + "def _(plt):\n" + expected + "    return\n"


def test_stock_marimo_moves_comments():
    """
    The reason the converter exists: stock `marimo convert` moves comments
    in these cells. If this fails, marimo has fixed the problem and the
    converter may no longer be needed.
    """
    stock = MarimoConvert.from_ipynb(make_notebook(CELLS)).to_py()
    ## count how many of the expected lines stock marimo gets right
    kept = 0
    for expected in EXPECTED_LINES:
        if expected in stock:
            kept += 1
        else:
            pass
    assert kept < len(EXPECTED_LINES)
