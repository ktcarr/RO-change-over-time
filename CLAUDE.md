# CLAUDE.md

Guidance for Claude Code in this repository. General preferences and the board conventions:
`~/claude-config` (loaded through `~/.claude/CLAUDE.md`). Path-scoped details:
`.claude/rules/ro.md` (the RO code: `src/ro.py`, `src/XRO.py`, `ro_tools`) and
`.claude/rules/notebook-conversion.md` (converting `.ipynb` to marimo).

## What this repo is

Code for the paper continuing chapter 3 of the author's PhD thesis: why ENSO variance changes
non-monotonically under warming in CESM2. The paper's claims:
1. El Niño and La Niña amplitudes change asymmetrically (El Niño starts decreasing ~50 years
   before La Niña).
2. The conceptual recharge oscillator (RO) model reproduces the CESM2 changes.
3. In the RO, the overall decrease and the asymmetry are driven largely by surface heat fluxes.

The repository will be published with the paper, and it also produced figures in the published
thesis (`~/thesis/thesis/chapters`: `intro.tex`, `chapter_enso-change.tex`). So reproducibility
and readability come first: prefer simple code to efficient code, so that someone new to the
project can follow every script. `README.md` has the layout and the key scripts in run order.

**Layout**:
- `scripts/cesm2/`: key CESM2 scripts (marimo notebooks saved as `.py`); `preprocess/` builds
  the indices and consolidated fields; `fig_scripts/` has one script per paper or thesis figure,
  named after its LaTeX label; `explore/` holds exploratory notebooks not needed to reproduce
  the paper or thesis.
- `src/`: the project's module (installed editable), by topic: `constants.py`, `data.py`,
  `indices.py`, `stats.py`, `plot.py`, `ro.py`. Put new reusable code in the matching module,
  not in per-figure helper files. `utils.py` holds older general utilities (to be moved into
  the topic modules).
- `tests/`: checks on known answers. `results/`: figures and intermediate results.
- Related: the paper's Overleaf project is `~/ch3` (live docs `paper-outline.tex`, `draft2.tex`);
  the RO package is `~/ro_tools`. Issues: this repo's tracker, on the "UH postdoc" board.

## Running (Casper)

- Python: `envs/bin/python` (the conda environment inside the repo). `DATA_FP` and `SAVE_FP`
  are set in `envs/conda-meta/state`; export them when calling `envs/bin/python` directly.
- marimo scripts: `marimo edit <script>.py` interactively, or headless with
  `MPLBACKEND=Agg envs/bin/python <script>.py`, run from `scripts/cesm2`.
- Run on the current Casper node by default. Submit PBS jobs (`qsub -A UHWM0049 -q casper`)
  only for parallel or heavy work, e.g. running many scripts or RO cases at once.
- Install packages with mamba, not pip:
  `/glade/u/apps/opt/conda/bin/mamba install -p ./envs -c conda-forge <pkg>`, and add them to
  `env.yml`. Dry-run first (`--dry-run`) and check what else it would change.
- Tests: `envs/bin/python -m pytest tests`.
- `gh` comes from the `gh` environment module (loaded in `~/.bashrc`; `module load gh` if
  `which gh` finds nothing).

## Conventions (this repo)

- Format with `envs/bin/black`.
- All imports at the top of the file, never inside functions (in marimo scripts, in the
  imports cell). Side-effect imports (e.g. `import cmocean` for its colormaps) get a comment.
- marimo: end a figure cell with `plt.gca()` (or `fig`), not `plt.show()`; one figure per cell.
