"""
Style checker for the repo's Python conventions (CLAUDE.md, "Conventions"),
run by Claude Code after every Edit/Write of a .py file (PostToolUse hook in
.claude/settings.json) and usable by hand.

Checks (each finding names its rule):
- docstring: every function has a docstring; functions with arguments list
  them under "Args:", functions that return a value have "Returns:" (test_*
  functions and marimo @app.cell functions are exempt from Args/Returns; cells
  also from the docstring)
- comments: no more than MAX_UNCOMMENTED logical steps in a row without a
  comment (a step is a simple statement or the header of if/for/while/with/
  try; `pass`, `else`, docstrings and the final `return` of a function don't
  count)
- none-first: `if x is not None:` (write the None branch first)
- early-return: an `if` without `else` whose body returns, followed by more
  statements (use if/else and one return)
- comprehension: a list/dict/set comprehension or generator spread over
  several lines (use an explicit loop)

As a hook it reads the tool call's JSON on stdin and reports only findings on
lines that differ from git HEAD (all lines of an untracked file), so older
code isn't flagged; findings go back to Claude as a PostToolUse "block"
decision (the edit stays; Claude is asked to fix them). By hand:
    ./envs/bin/python .claude/hooks/check_style.py [--all] FILE [FILE ...]
(--all: every line, not only the changed ones). Exit code 1 if anything is
found by hand; the hook always exits 0 (the JSON carries the result).

Standard library only, so any Python >= 3.8 runs it.
"""

import argparse
import ast
import io
import json
import pathlib
import re
import subprocess
import sys
import tokenize

## most logical steps in a row without a comment (CLAUDE.md: a comment every
## 1-3 lines)
MAX_UNCOMMENTED = 3

## path parts never checked (environments, generated marimo sessions)
SKIP_PARTS = {"envs", "__marimo__", ".venv", "site-packages"}


def comment_lines(source):
    """
    Line numbers that hold a comment (full-line or inline).

    Args:
        - source (str): file contents

    Returns:
        - lines (set of int): 1-based line numbers with a "#" comment token
    """
    lines = set()
    ## tokenize, so "#" inside strings isn't taken for a comment
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for token in tokens:
        if token.type == tokenize.COMMENT:
            lines.add(token.start[0])
        else:
            pass
    return lines


def is_docstring(stmt):
    """
    Whether a statement is a bare string literal (a docstring).

    Args:
        - stmt (ast.stmt): statement

    Returns:
        - result (bool)
    """
    ## an expression statement whose value is a str constant
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        result = isinstance(stmt.value.value, str)
    else:
        result = False
    return result


def is_marimo_cell(func):
    """
    Whether a function is a marimo cell (decorated with app.cell or similar).

    Args:
        - func (ast.FunctionDef): function

    Returns:
        - result (bool)
    """
    result = False
    for decorator in func.decorator_list:
        ## @app.cell or @app.cell(hide_code=True)
        if isinstance(decorator, ast.Call):
            target = decorator.func
        else:
            target = decorator
        if isinstance(target, ast.Attribute) and target.attr in ("cell", "function"):
            result = True
        else:
            pass
    return result


def header_end(stmt):
    """
    Last line of a statement's own text: the whole statement if simple, the
    line before its body if compound (if/for/while/with/try/def).

    Args:
        - stmt (ast.stmt): statement

    Returns:
        - line (int): 1-based line number
    """
    body = getattr(stmt, "body", None)
    if isinstance(body, list) and len(body) > 0:
        ## compound: the header ends before the first body statement
        line = max(stmt.lineno, body[0].lineno - 1)
    else:
        line = stmt.end_lineno
    return line


def covered_by_comment(first, last, comments):
    """
    Whether any comment line lies in [first, last].

    Args:
        - first (int): first line of the window
        - last (int): last line of the window
        - comments (set of int): comment line numbers (comment_lines)

    Returns:
        - covered (bool)
    """
    covered = False
    for line in range(first, last + 1):
        if line in comments:
            covered = True
        else:
            pass
    return covered


def sub_blocks(stmt):
    """
    The nested blocks of a compound statement, each with the line after
    which its comments may start (the line before its first statement's
    window).

    Args:
        - stmt (ast.stmt): statement

    Returns:
        - blocks (list of tuple): (statements, previous line) per block:
          body, else/elif, except handlers, finally
    """
    blocks = []
    ## the main body follows the header
    body = getattr(stmt, "body", [])
    if isinstance(body, list) and len(body) > 0:
        blocks.append((body, header_end(stmt)))
        last_end = body[-1].end_lineno
    else:
        last_end = stmt.end_lineno
    ## except handlers start after the body (or the previous handler)
    for handler in getattr(stmt, "handlers", []):
        blocks.append((handler.body, handler.lineno))
        last_end = handler.body[-1].end_lineno
    ## else/elif and finally: comments may sit just above their keyword
    for field in ("orelse", "finalbody"):
        block = getattr(stmt, field, [])
        if isinstance(block, list) and len(block) > 0:
            blocks.append((block, last_end))
            last_end = block[-1].end_lineno
        else:
            pass
    return blocks


def check_block(block, previous_end, comments, final, findings):
    """
    Runs of more than MAX_UNCOMMENTED uncommented steps among the
    statements of one block, then the same in each nested block. A compound
    statement is one step of its block (a comment above it covers its
    header); its body is counted as a block of its own.

    Args:
        - block (list of ast.stmt): statements of one block
        - previous_end (int): line after which comments for the block's
          first statement may start
        - comments (set of int): comment line numbers (comment_lines)
        - final (ast.stmt or None): the function's final return (not a step)
        - findings (list of tuple): (first line, last line, message), added to

    Returns:
        - None (appends to findings)
    """
    run = []
    for stmt in block:
        ## nested functions and classes are checked on their own
        nested_def = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        if isinstance(stmt, nested_def):
            previous_end = stmt.end_lineno
            run = []
            continue
        else:
            pass

        ## pass, docstrings and the final return don't count as steps
        trivial = isinstance(stmt, ast.Pass) or is_docstring(stmt) or stmt is final
        covered = covered_by_comment(previous_end + 1, header_end(stmt), comments)
        if trivial:
            pass
        elif covered:
            run = [stmt]
        else:
            run.append(stmt)

        ## report a run once, when it first gets too long
        if not trivial and len(run) == MAX_UNCOMMENTED + 1:
            first = run[0].lineno
            last = header_end(stmt)
            ## the run's span, so the hook can match it to changed lines
            message = (
                f"comments: {MAX_UNCOMMENTED + 1}+ steps without a comment "
                f"(lines {first}-{last})"
            )
            findings.append((first, last, message))
        else:
            pass

        ## the nested blocks (if/else bodies, loops, try), counted on their own
        for sub_block, sub_previous in sub_blocks(stmt):
            check_block(sub_block, sub_previous, comments, final, findings)
        previous_end = stmt.end_lineno
    return None


def check_comments(func, comments):
    """
    Runs of more than MAX_UNCOMMENTED steps without a comment, block by
    block (check_block).

    Args:
        - func (ast.FunctionDef): function
        - comments (set of int): comment line numbers (comment_lines)

    Returns:
        - findings (list of tuple): (first line, last line, message) per run
    """
    findings = []
    ## the final return of the function isn't a step of its own
    if len(func.body) > 0 and isinstance(func.body[-1], ast.Return):
        final = func.body[-1]
    else:
        final = None
    ## the function body, from the line after the def's header
    check_block(func.body, header_end(func), comments, final, findings)
    return findings


def check_docstring(func):
    """
    Docstring present, with Args:/Returns: where needed.

    Args:
        - func (ast.FunctionDef): function

    Returns:
        - findings (list of tuple): (line, line, message)
    """
    findings = []
    docstring = ast.get_docstring(func)

    ## exemptions: marimo cells (no docstring needed), tests (no Args/Returns)
    cell = is_marimo_cell(func)
    test = func.name.startswith("test_")

    ## arguments other than self/cls
    names = []
    for arg in func.args.posonlyargs + func.args.args + func.args.kwonlyargs:
        if arg.arg in ("self", "cls"):
            pass
        else:
            names.append(arg.arg)

    ## a return with a value anywhere in the function itself
    returns_value = False
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            returns_value = True
        else:
            pass

    ## marimo cells need none; others need one, with Args/Returns unless tests
    if cell:
        pass
    elif docstring is None:
        findings.append((func.lineno, func.lineno, f"docstring: {func.name} has none"))
    elif test:
        pass
    else:
        if len(names) > 0 and "Args:" not in docstring:
            findings.append(
                (func.lineno, func.lineno, f"docstring: {func.name} lacks Args:")
            )
        else:
            pass
        if returns_value and "Returns:" not in docstring:
            findings.append(
                (func.lineno, func.lineno, f"docstring: {func.name} lacks Returns:")
            )
        else:
            pass
    return findings


def check_statements(func):
    """
    `is not None` tests, early returns and multi-line comprehensions.

    Args:
        - func (ast.FunctionDef): function

    Returns:
        - findings (list of tuple): (line, line, message)
    """
    findings = []
    for node in ast.walk(func):
        ## if x is not None:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            test = node.test
            ## a single `is not` comparison ...
            is_not = len(test.ops) == 1 and isinstance(test.ops[0], ast.IsNot)
            ## ... against the constant None
            none = isinstance(test.comparators[0], ast.Constant)
            none = none and test.comparators[0].value is None
            if is_not and none:
                findings.append(
                    (
                        node.lineno,
                        node.lineno,
                        "none-first: `is not None`; put the None branch first",
                    )
                )
            else:
                pass
        else:
            pass

        ## multi-line comprehensions
        comprehension_types = (
            ast.ListComp,
            ast.DictComp,
            ast.SetComp,
            ast.GeneratorExp,
        )
        if isinstance(node, comprehension_types) and node.end_lineno > node.lineno:
            findings.append(
                (
                    node.lineno,
                    node.end_lineno,
                    "comprehension: spans several lines; use a loop",
                )
            )
        else:
            pass

    ## early return: an if without else ending in return, with statements after
    blocks = [func.body]
    for node in ast.walk(func):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list) and node is not func:
                blocks.append(block)
            else:
                pass
    for block in blocks:
        for index, stmt in enumerate(block[:-1]):
            early = isinstance(stmt, ast.If) and len(stmt.orelse) == 0
            early = early and isinstance(stmt.body[-1], ast.Return)
            if early:
                findings.append(
                    (
                        stmt.lineno,
                        stmt.lineno,
                        "early-return: `if` without `else` returns early",
                    )
                )
            else:
                pass
    return findings


def check_source(source):
    """
    Every finding in a file's source.

    Args:
        - source (str): file contents

    Returns:
        - findings (list of tuple): (first line, last line, message), sorted
    """
    tree = ast.parse(source)
    comments = comment_lines(source)
    findings = []
    ## every function and method, nested ones included
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            findings.extend(check_docstring(node))
            findings.extend(check_comments(node, comments))
            findings.extend(check_statements(node))
        else:
            pass
    findings.sort()
    return findings


def changed_lines(fp):
    """
    Lines of a file that differ from git HEAD.

    Args:
        - fp (pathlib.Path): file inside a git work tree

    Returns:
        - lines (set of int or None): changed 1-based lines; None means every
          line (untracked file, or no git)
    """
    ## untracked (or outside git): treat the whole file as new
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", fp.name],
        cwd=fp.parent,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        lines = None
    else:
        ## hunk headers "@@ -a,b +c,d @@" of a zero-context diff: lines c..c+d-1
        diff = subprocess.run(
            ["git", "diff", "-U0", "HEAD", "--", fp.name],
            cwd=fp.parent,
            capture_output=True,
            text=True,
        )
        lines = set()
        for match in re.finditer(r"^@@ .* \+(\d+)(?:,(\d+))? @@", diff.stdout, re.M):
            start = int(match.group(1))
            count = 1 if match.group(2) is None else int(match.group(2))
            lines.update(range(start, start + count))
    return lines


def check_file(fp, only_changed):
    """
    Findings in one file, optionally only those touching changed lines.

    Args:
        - fp (pathlib.Path): Python file
        - only_changed (bool): keep only findings on lines changed since HEAD

    Returns:
        - findings (list of tuple): (first line, last line, message)
    """
    source = fp.read_text()
    ## a file that doesn't parse (mid-edit) has nothing to check yet
    try:
        findings = check_source(source)
    except SyntaxError:
        findings = []

    ## keep findings whose line span overlaps a changed line
    if only_changed:
        changed = changed_lines(fp)
    else:
        changed = None
    if changed is None:
        kept = findings
    else:
        kept = []
        for first, last, message in findings:
            if changed.intersection(range(first, last + 1)):
                kept.append((first, last, message))
            else:
                pass
    return kept


def skipped(fp):
    """
    Whether a path is outside the checks (not .py, or in SKIP_PARTS).

    Args:
        - fp (pathlib.Path): path

    Returns:
        - result (bool)
    """
    result = fp.suffix != ".py" or len(SKIP_PARTS.intersection(fp.parts)) > 0
    return result


def run_hook():
    """
    PostToolUse hook: check the edited file, report findings to Claude.

    Returns:
        - None (prints the hook's JSON when there are findings)
    """
    ## the tool call (Edit or Write), from stdin
    payload = json.load(sys.stdin)
    tool_input = payload.get("tool_input", {})

    ## the edited file (Edit and Write both pass file_path)
    fp_str = tool_input.get("file_path", "")
    fp = pathlib.Path(fp_str)

    ## only existing Python files outside the skipped directories
    if fp_str == "" or skipped(fp) or not fp.is_file():
        findings = []
    else:
        findings = check_file(fp, only_changed=True)

    ## findings back to Claude (the edit stays; Claude is asked to fix them)
    if len(findings) == 0:
        pass
    else:
        ## one line per finding: file:line: rule: message
        lines = []
        for first, _, message in findings:
            lines.append(f"  {fp.name}:{first}: {message}")
        ## PostToolUse "block": the reason is shown to Claude, the edit stays
        reason = (
            "Style check (CLAUDE.md conventions) on the changed lines of "
            f"{fp}:\n" + "\n".join(lines) + "\nFix these, or say why one doesn't apply."
        )
        print(json.dumps({"decision": "block", "reason": reason}))
    return None


def main():
    """
    By hand: check files given on the command line; without files, run as
    the hook (stdin JSON).

    Returns:
        - None (exits 1 if anything was found by hand)
    """
    ## files to check by hand (none: hook mode); --all for every line
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="*", type=pathlib.Path)
    parser.add_argument("--all", action="store_true", help="check every line")

    ## parsed from sys.argv (the hook passes none)
    args = parser.parse_args()

    ## no files: hook mode
    if len(args.files) == 0:
        run_hook()
        status = 0
    else:
        ## each file, all lines or only the changed ones
        status = 0
        for fp in args.files:
            for first, _, message in check_file(fp, only_changed=not args.all):
                print(f"{fp}:{first}: {message}")
                status = 1
    sys.exit(status)


if __name__ == "__main__":
    main()
