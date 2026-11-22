"""
Known-answer tests for src/plot.py.
"""

import matplotlib

## non-interactive backend, so the tests run without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import src.plot


def test_make_centered_levels():
    """Levels are symmetric, evenly spaced, with a band centered on zero."""

    ## dx=1, two levels per side
    levels = src.plot.make_centered_levels(dx=1, nlevs=2)
    np.testing.assert_allclose(levels, [-1.5, -0.5, 0.5, 1.5])

    ## a non-integer spacing gives the same pattern
    levels = src.plot.make_centered_levels(dx=0.4, nlevs=5)

    ## 10 levels, the first positive one at dx/2, evenly spaced
    assert len(levels) == 10
    np.testing.assert_allclose(levels[5], 0.2)
    np.testing.assert_allclose(np.diff(levels), 0.4)


def test_save_fig(tmp_path, monkeypatch):
    """save_fig writes SAVE_FP/<subdir>/<name>.pdf, creating the folder."""

    ## point SAVE_FP at a temporary folder
    monkeypatch.setenv("SAVE_FP", str(tmp_path))

    ## save a small figure
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    ## save it to a subfolder that doesn't exist yet
    fp = src.plot.save_fig(fig, "test-fig", subdir="figs")

    ## free the figure's memory
    plt.close(fig)

    ## the file exists where expected and is a PDF
    assert fp == tmp_path / "figs" / "test-fig.pdf"
    assert fp.read_bytes()[:4] == b"%PDF"
