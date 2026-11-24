---
paths:
  - "src/ro.py"
  - "src/XRO.py"
  - "src/XRO_utils.py"
  - "scripts/cesm2/fig_scripts/RO-*.py"
  - "scripts/cesm2/fig_scripts/gr-over-time.py"
  - "scripts/cesm2/ro_sensitivity/**"
  - "scripts/cesm2/alpha-over-time.py"
---

## The RO code (details)

- `src/ro.py` holds the RO fit and simulation steps shared by the RO figures. It uses the RO
  model from `~/ro_tools` (`ro_tools.xro.XRO`), installed editable in `envs/` without
  dependencies (`pip install --no-deps -e ~/ro_tools`; they come from `env.yml`).
- `src/XRO.py` is the old RO implementation. With the current settings `ro_tools` reproduces it
  to roundoff. It stays only for the scripts in `explore/` and `src/utils.py`; don't use it in
  new code.
- Changes to the fitting itself belong in `ro_tools` (issues in `ktcarr/ro_tools`), not here.
