---
paths:
  - "**/*.ipynb"
  - "src/convert_marimo.py"
  - "tests/test_convert_marimo.py"
---

## Converting notebooks to marimo (details)

- Never use `marimo convert` as-is. Its fix for variables defined in several cells rewrites the
  cell through `ast.unparse`, which drops comments. Use `src/convert_marimo.py`, which makes
  the same fixes as text edits and checks that every comment survives.
- Code converted from older notebooks doesn't meet the style rules yet; bring it up to them
  when restructuring a script, not during conversion.
