# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

Code for the paper continuing chapter 3 of the author's PhD thesis: why ENSO variance changes
non-monotonically under warming in CESM2. The paper's claims:
1. El Niño and La Niña amplitudes change asymmetrically (El Niño starts decreasing ~50 years
   before La Niña).
2. The conceptual recharge oscillator (RO) model reproduces the CESM2 changes.
3. In the RO, the overall decrease and the asymmetry are driven largely by surface heat fluxes.

The repository will be published with the paper, and it also produced figures in the published
thesis (`~/thesis/thesis/chapters`: `intro.tex`, `chapter_enso-change.tex`). See `README.md` for
the layout and the key scripts, in the order they run.

- `scripts/cesm2/`: key CESM2 scripts (marimo notebooks saved as `.py`); `preprocess/` builds
  the indices and consolidated fields; `explore/` holds exploratory Jupyter notebooks not needed
  to reproduce the paper or thesis.
- `src/`: the project's module (installed editable); `src/XRO.py` is the old RO implementation,
  to be replaced by `~/ro_tools`.
- `tests/`: checks on known answers. `results/`: figures and intermediate results.
- Related: the paper's Overleaf project is `~/ch3` (live docs `paper-outline.tex`, `draft2.tex`);
  the improved RO package is `~/ro_tools`.

## Principles

- Reproducibility and interpretability come first. Prefer simple, readable code to efficient
  code. Someone new to the project should be able to follow every script.
- Comment generously (see Conventions); the style hook enforces a baseline.
- Test changes, and keep the tests in `tests/`.
- Work one step at a time: test, then stop for the user's review before the next step.
  Commit or push only when asked.

## Running

- Python: `envs/bin/python` (the conda environment inside the repo). `DATA_FP` and `SAVE_FP`
  are set in `envs/conda-meta/state`; export them when calling `envs/bin/python` directly.
- marimo scripts: `marimo edit <script>.py` interactively, or headless with
  `MPLBACKEND=Agg envs/bin/python <script>.py`, run from `scripts/cesm2`.
- Run on the current Casper node by default. Submit PBS jobs (`qsub -A UHWM0049 -q casper`)
  only for parallel or heavy work, e.g. running many scripts at once.
- Install packages with mamba, not pip:
  `/glade/u/apps/opt/conda/bin/mamba install -p ./envs -c conda-forge <pkg>`, and add them to
  `env.yml`. Dry-run first (`--dry-run`) and check what else it would change.
- Tests: `envs/bin/python -m pytest tests`.

## Conventions

- Every function gets a docstring (purpose, typed args, typed returns). Comment generously:
  a comment before each logical step (every 1-3 lines), saying what it does and why.
- Prefer explicit loops to comprehensions that need a line break. No early `return` inside
  an `if` without an `else`: use `if`/`else` and return once. Write `if x is None: ... else: ...`,
  not `if x is not None`.
- Format with `envs/bin/black`.
- marimo: end a figure cell with `plt.gca()` (or `fig`), not `plt.show()`; one figure per cell.
- Converting notebooks: never use `marimo convert` as-is. Its fix for variables defined in
  several cells rewrites the cell through `ast.unparse`, which drops comments.
- A PostToolUse hook (`.claude/settings.json`) runs `.claude/hooks/check_style.py` after every
  Edit/Write of a `.py` file and reports findings on the lines changed since `HEAD`. By hand:
  `python3 .claude/hooks/check_style.py [--all] FILE`. The script is a copy of the one in
  `~/ro_tools` (itself copied from `dlesym-pacemaker_intermodel`); keep the copies the same.
  Code converted from older notebooks doesn't meet these rules yet; bring it up to them when
  restructuring a script, not during conversion.
