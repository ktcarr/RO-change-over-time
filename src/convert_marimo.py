"""
Convert a Jupyter notebook (.ipynb) to a marimo notebook (.py) without losing
comments.

Why this exists: `marimo convert` (marimo 0.25) fixes cells that marimo would
reject (a variable defined in several cells, `x += 1` on another cell's
variable, an import repeated in a later cell) by editing the cell's syntax
tree and writing it back with `ast.unparse`. `ast.unparse` drops every comment
and all formatting in that cell; marimo's `CommentPreserver` then re-inserts
some comments by line number, often on the wrong line or several on one line.

What this does: run marimo's own conversion, but with the three helpers that
rewrite cells replaced by versions that make the same changes as small text
edits at the positions the syntax tree records. Everything else in the cell
(comments, blank lines, formatting) stays exactly as written. marimo's
`CommentPreserver` is switched off, since there is nothing left to restore.
After converting, every comment of the notebook's code cells is checked to be
in the output (`check_comments`).

One addition of our own: marimo displays a cell's last expression, not
`plt.show()`, so figure cells are made to end with `plt.gca()`
(`end_figure_cells`). Cells drawing several figures (a `plt.show()` before
the end) are left as they are.

The replaced helpers, all in `marimo._convert.ipynb.to_ir`:
- `_rename_in_scope`: renames variables (e.g. `fig` -> `_fig`, `x` -> `x_1`)
- `_transform_aug_assign`: expands `x += 1` to `x = x + 1` before renaming
- `transform_remove_duplicate_imports`: drops imports an earlier cell made

Usage (from the repository root):
    envs/bin/python -m src.convert_marimo NOTEBOOK.ipynb OUTPUT.py
"""

import argparse
import ast
import logging
import copy
import io
import json
import pathlib
import re
import tokenize
import unittest.mock

import marimo._convert.common.comment_preserver as comment_preserver
import marimo._convert.ipynb.to_ir as to_ir
from marimo._ast.compiler import compile_cell
from marimo._ast.transformers import RemoveImportTransformer
from marimo._ast.visitor import ScopedVisitor
from marimo._convert.converters import MarimoConvert
from marimo._types.ids import CellId_t

## operator symbols for expanding augmented assignments (x += 1 -> x = x + 1)
OPERATORS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.MatMult: "@",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
}

## expressions that never need parentheses on the right of an operator
ATOMIC = (ast.Name, ast.Constant, ast.Attribute, ast.Subscript, ast.Call)


def line_offsets(source):
    """
    Character offset at which each line of a source string starts.

    Args:
        - source (str): source code

    Returns:
        - offsets (list of int): offsets[i] is where line i + 1 starts
    """
    offsets = [0]
    ## keepends, so each line's length includes its newline
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def char_col(source_lines, lineno, byte_col):
    """
    Convert an AST column (a UTF-8 byte offset) to a character offset. They
    differ on lines with non-ASCII text before the column (e.g. "Niño").

    Args:
        - source_lines (list of str): source split into lines
        - lineno (int): 1-based line number
        - byte_col (int): column as the AST gives it (bytes)

    Returns:
        - col (int): column in characters
    """
    ## decode the line's first byte_col bytes and count the characters
    line_bytes = source_lines[lineno - 1].encode("utf-8")
    col = len(line_bytes[:byte_col].decode("utf-8"))
    return col


def apply_edits(source, edits):
    """
    Replace spans of a source string with new text. No span may touch a
    comment: that is what guarantees every comment stays where it was.

    Args:
        - source (str): source code
        - edits (list of tuple): (start line, start col, end line, end col,
          new text) per edit; lines 1-based, columns in characters. Spans
          must not overlap.

    Returns:
        - new_source (str): source with every edit applied
    """
    offsets = line_offsets(source)

    ## absolute character spans of the comments (tokenize columns are chars);
    ## only needed if there are edits (cells that don't parse get none, and
    ## may not tokenize either)
    comments = []
    if len(edits) == 0:
        tokens = []
    else:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for token in tokens:
        if token.type == tokenize.COMMENT:
            comment_start = offsets[token.start[0] - 1] + token.start[1]
            comments.append((comment_start, comment_start + len(token.string)))
        else:
            pass

    ## absolute character positions of each span
    spans = []
    for start_line, start_col, end_line, end_col, text in edits:
        start = offsets[start_line - 1] + start_col
        end = offsets[end_line - 1] + end_col
        ## refuse any edit overlapping a comment
        for comment_start, comment_end in comments:
            if start < comment_end and comment_start < end:
                raise ValueError(f"edit would change a comment:\n{source}")
            else:
                pass
        spans.append((start, end, text))

    ## apply from the end, so earlier positions stay valid
    new_source = source
    for start, end, text in sorted(spans, reverse=True):
        new_source = new_source[:start] + text + new_source[end:]
    return new_source


def node_name(node):
    """
    The name a renamable node binds or refers to (marimo's NamedNode types).

    Args:
        - node (ast.AST): any node

    Returns:
        - name (str or None): the name; None for nodes without one
    """
    ## variables, then functions/classes; marimo's renamer handles no others
    ## that appear in these notebooks (match patterns, type parameters)
    if isinstance(node, ast.Name):
        name = node.id
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        name = node.name
    else:
        name = None
    return name


def name_span(node, name, source_lines):
    """
    Where a node's name is written in the source.

    Args:
        - node (ast.AST): a Name, FunctionDef, AsyncFunctionDef or ClassDef
        - name (str): the name as written in the source (before renaming)
        - source_lines (list of str): source split into lines

    Returns:
        - span (tuple): (line, start col, end col), columns in characters
    """
    line = node.lineno
    start = char_col(source_lines, line, node.col_offset)
    if isinstance(node, ast.Name):
        ## a Name's own span is exactly the identifier
        end = char_col(source_lines, line, node.end_col_offset)
    else:
        ## for def/class the node starts at the keyword: find the name after it
        pattern = r"(?:async\s+)?(?:def|class)\s+(" + re.escape(name) + ")"
        match = re.compile(pattern).match(source_lines[line - 1], start)
        ## group 1 is the name itself
        start = match.start(1)
        end = match.end(1)
    return (line, start, end)


def rename_in_scope_text(source, cell_idx, renamer):
    """
    Text-editing replacement for marimo's `_rename_in_scope`: rename
    top-level names in one cell, leaving the rest of the text untouched.

    Args:
        - source (str): the cell's source
        - cell_idx (int): the cell's index in the notebook
        - renamer (to_ir.Renamer): marimo's renamer for this notebook

    Returns:
        - new_source (str): the cell with names renamed
    """
    ## cells that don't parse are left alone, as in marimo
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None

    if tree is None:
        new_source = source
    else:
        ## each named node's name before renaming (the renamer edits in place)
        originals = {}
        for node in ast.walk(tree):
            originals[id(node)] = node_name(node)

        ## marimo's renaming rules, copied from to_ir._rename_in_scope: rename
        ## a definition only if the cell's top level defines the name and no
        ## inner scope (a function argument, a comprehension) rebinds it
        def on_def(node, name, block_stack):
            """
            Rename a definition, following marimo's scoping rule.

            Args:
                - node (ast.AST): the defining node
                - name (str): the defined name
                - block_stack (list of marimo Block): enclosing scopes

            Returns:
                - None (renames node in place)
            """
            ## an inner scope rebinding the name keeps its own
            rebound = False
            for block in block_stack[1:]:
                if block.is_defined(name):
                    rebound = True
                else:
                    pass
            ## `global x` inside a function, or a top-level definition
            is_global = name in block_stack[-1].global_names
            if is_global or (block_stack[0].is_defined(name) and not rebound):
                renamer.rename_named_node(cell_idx, node, is_reference=False)
            else:
                pass
            return None

        def on_ref(node):
            """
            Rename a reference (marimo resolves it to the right definition).

            Args:
                - node (ast.AST): the referring node

            Returns:
                - None (renames node in place)
            """
            renamer.rename_named_node(cell_idx, node, is_reference=True)
            return None

        ## run marimo's visitor; it raises on constructs marimo rejects
        visitor = ScopedVisitor(ignore_local=True, on_def=on_def, on_ref=on_ref)
        try:
            visitor.visit(tree)
            visited = True
        except SyntaxError:
            visited = False

        ## one text edit per node whose name the renamer changed
        source_lines = source.splitlines()
        edits = []
        if visited:
            for node in ast.walk(tree):
                ## the name before and after the renamer ran
                old = originals.get(id(node))
                new = node_name(node)
                ## unnamed nodes and unchanged names need no edit
                if old is None or old == new:
                    pass
                else:
                    line, start, end = name_span(node, old, source_lines)
                    edits.append((line, start, line, end, new))
        else:
            pass
        ## renamed cell (unchanged if no edits)
        new_source = apply_edits(source, edits)
    return new_source


def transform_aug_assign_text(sources):
    """
    Text-editing replacement for marimo's `_transform_aug_assign`: expand
    `x += value` to `x = x + value`, so renaming can give the new `x` its own
    name while the right side still reads the previous cell's `x`. Only
    plain-variable targets are expanded: `ds["a"] += 1` defines nothing.

    Args:
        - sources (list of str): every cell's source

    Returns:
        - new_sources (list of str): sources with augmented assignments expanded
    """
    new_sources = []
    for source in sources:
        ## cells that don't parse are left alone
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None

        ## text edits for this cell (none if it doesn't parse)
        edits = []
        if tree is None:
            pass
        else:
            source_lines = source.splitlines()
            for node in ast.walk(tree):
                ## x += value, with x a plain variable
                is_aug = isinstance(node, ast.AugAssign)
                if is_aug and isinstance(node.target, ast.Name):
                    ## the pieces of "x = x + value"
                    target = node.target.id
                    symbol = OPERATORS[type(node.op)]
                    value = node.value

                    ## the operator ("+=") follows the target on its line
                    line0 = node.target.end_lineno
                    text0 = source_lines[line0 - 1]
                    col0 = char_col(source_lines, line0, node.target.end_col_offset)
                    ## search for it from the end of the target
                    op_start = text0.find(symbol + "=", col0)
                    ## the operator's end, after its "="
                    op_end = op_start + len(symbol) + 1
                    ## (only "x \\\n += 1" would put it on the next line)
                    if op_start == -1:
                        raise ValueError(f"augmented assignment not found:\n{source}")
                    else:
                        pass

                    ## replace just the operator: "x += v" -> "x = x + v"
                    new_op = f"= {target} {symbol}"
                    edits.append((line0, op_start, line0, op_end, new_op))

                    ## text between the operator and where the value starts;
                    ## a "(" there means the source already wraps the whole
                    ## value in parentheses (ast positions exclude them)
                    line1 = value.lineno
                    col1 = char_col(source_lines, line1, value.col_offset)
                    ## a value starting on a later line: the rest of this one
                    if line1 == line0:
                        between = text0[op_end:col1]
                    else:
                        between = text0[op_end:]
                    wrapped = "(" in between

                    ## otherwise parenthesize values that could bind looser
                    ## than the operator (x -= a + b -> x = x - (a + b))
                    if isinstance(value, ATOMIC) or wrapped:
                        pass
                    else:
                        line2 = value.end_lineno
                        col2 = char_col(source_lines, line2, value.end_col_offset)
                        ## "(" at the value's start, ")" at its end
                        edits.append((line1, col1, line1, col1, "("))
                        edits.append((line2, col2, line2, col2, ")"))
                else:
                    pass
        new_sources.append(apply_edits(source, edits))
    return new_sources


def strip_imports_text(source, transformer):
    """
    Text-editing replacement for `RemoveImportTransformer.strip_imports`:
    remove one name from the cell's import statements, editing only those
    statements.

    Args:
        - source (str): the cell's source
        - transformer (RemoveImportTransformer): marimo's transformer for the
          name (it decides which statements lose the name; keep_one keeps the
          first)

    Returns:
        - new_source (str): the cell without the duplicate import
    """
    tree = ast.parse(source)
    source_lines = source.splitlines()

    ## the parent block of every statement, to know when a block would empty
    parents = {}
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list):
                for stmt in block:
                    parents[id(stmt)] = block
            else:
                pass

    ## import statements in source order (the transformer's keep_one keeps
    ## the first match, so order matters)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        else:
            pass
    imports.sort(key=lambda node: (node.lineno, node.col_offset))

    ## one edit per import statement the transformer changes
    edits = []
    for node in imports:
        ## what marimo's transformer makes of this statement on its own
        result = transformer.visit(copy.deepcopy(node))
        ## the statement's span in the source
        start_line = node.lineno
        start = char_col(source_lines, start_line, node.col_offset)
        ## ... to its end
        end_line = node.end_lineno
        end = char_col(source_lines, end_line, node.end_col_offset)

        if result is None:
            ## statement removed: a block can't be empty, so leave `pass` there
            if len(parents[id(node)]) == 1:
                edits.append((start_line, start, end_line, end, "pass"))
            else:
                ## also remove the spaces after it, so a trailing comment
                ## moves to the statement's indentation
                rest = source_lines[end_line - 1][end:]
                end = end + len(rest) - len(rest.lstrip(" \t"))
                edits.append((start_line, start, end_line, end, ""))
        elif len(result.names) != len(node.names):
            ## some names removed: rewrite just this statement
            edits.append((start_line, start, end_line, end, ast.unparse(result)))
        else:
            pass
    new_source = apply_edits(source, edits)

    ## drop the lines left blank by removed statements
    if new_source == source:
        pass
    else:
        kept = []
        old_lines = set(source.splitlines())
        for line in new_source.split("\n"):
            ## an emptied line: blank now, and not one of the original lines
            emptied = line.strip() == "" and line not in old_lines
            if emptied:
                pass
            else:
                kept.append(line)
        new_source = "\n".join(kept)
    return new_source


def transform_remove_duplicate_imports_text(sources):
    """
    Replacement for marimo's `transform_remove_duplicate_imports`: identical
    except that it removes imports with `strip_imports_text`.

    Args:
        - sources (list of str): every cell's source

    Returns:
        - new_sources (list of str): sources without imports an earlier cell
          (or the same cell) already made
    """
    imports = set()
    new_sources = []
    for source in sources:
        ## cells that don't compile are left alone, as in marimo
        try:
            cell = compile_cell(source, cell_id=CellId_t("temp"))
        except SyntaxError:
            cell = None

        if cell is None:
            new_sources.append(source)
        else:
            ## names this cell imports; repeats of earlier imports are removed
            scoped = set()
            for var, instances in cell.variable_data.items():
                for instance in instances:
                    ## an import of a name already imported (anywhere so far)
                    repeated = var in imports or var in scoped
                    if repeated and instance.kind == "import":
                        ## keep one copy if only this cell imports it twice
                        keep_one = var not in imports
                        transformer = RemoveImportTransformer(var, keep_one=keep_one)
                        source = strip_imports_text(source, transformer)
                    else:
                        pass
                    scoped.add(var)
            ## later cells treat this cell's imports as already made
            imports.update(scoped)
            new_sources.append(source)
    return new_sources


def is_plt_call(node, attributes):
    """
    Whether a node is a call `plt.<attribute>(...)`.

    Args:
        - node (ast.AST): any node
        - attributes (set of str): accepted attributes, e.g. {"show"}

    Returns:
        - result (bool)
    """
    ## a call of an attribute of the name `plt`
    result = isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    result = result and isinstance(node.func.value, ast.Name)
    result = result and node.func.value.id == "plt" and node.func.attr in attributes
    return result


def top_level_nodes(statements):
    """
    Every node in a list of statements, not descending into functions or
    classes they define (whose code doesn't run when the cell runs).

    Args:
        - statements (list of ast.stmt): e.g. a cell function's body

    Returns:
        - nodes (list of ast.AST)
    """
    nodes = []
    ## depth-first through the statements, skipping nested definitions
    stack = list(statements)
    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    while len(stack) > 0:
        node = stack.pop()
        if isinstance(node, definitions):
            pass
        else:
            nodes.append(node)
            stack.extend(ast.iter_child_nodes(node))
    return nodes


def end_figure_cells(marimo_text):
    """
    Make each figure cell end with `plt.gca()`, which marimo displays:
    - a final `plt.show()` becomes `plt.gca()`;
    - a cell that makes a figure (plt.subplots/plt.figure) without calling
      plt.show() gets `plt.gca()` added before its return.
    Cells that call plt.show() before the end (several figures) are left
    alone: they need splitting into one cell per figure.

    Args:
        - marimo_text (str): a marimo notebook

    Returns:
        - new_text (str): the notebook with figure cells ending in plt.gca()
    """
    tree = ast.parse(marimo_text)
    edits = []
    for node in tree.body:
        ## cell functions: @app.cell or @app.cell(hide_code=True)
        if isinstance(node, ast.FunctionDef):
            cell = is_marimo_cell(node)
        else:
            cell = False

        ## the last statement before the cell's `return`
        if cell and len(node.body) >= 2 and isinstance(node.body[-1], ast.Return):
            last = node.body[-2]
            ret = node.body[-1]
        else:
            last = None

        if last is None:
            pass
        else:
            ## what the cell's own code does (not functions it defines)
            nodes = top_level_nodes(node.body)
            ## does it make a figure, and does it ever show one?
            makes_figure = False
            shows = False
            for child in nodes:
                makes_figure = makes_figure or is_plt_call(
                    child, {"subplots", "figure"}
                )
                shows = shows or is_plt_call(child, {"show"})
            ## does it already end with plt.show() or plt.gca()?
            ends_show = isinstance(last, ast.Expr) and is_plt_call(last.value, {"show"})
            ends_gca = isinstance(last, ast.Expr) and is_plt_call(last.value, {"gca"})

            ## final plt.show() -> plt.gca() (same columns: it's ASCII-only)
            if ends_show:
                call = last.value
                edits.append(
                    (
                        call.lineno,
                        call.col_offset,
                        call.end_lineno,
                        call.end_col_offset,
                        "plt.gca()",
                    )
                )
            ## a figure never shown: add plt.gca() above the return
            elif makes_figure and not shows and not ends_gca:
                indent = " " * ret.col_offset
                edits.append((ret.lineno, 0, ret.lineno, 0, f"{indent}plt.gca()\n"))
            else:
                pass
    new_text = apply_edits(marimo_text, edits)
    return new_text


def is_marimo_cell(func):
    """
    Whether a function is a marimo cell (decorated @app.cell, with or
    without arguments).

    Args:
        - func (ast.FunctionDef): function

    Returns:
        - result (bool)
    """
    result = False
    for decorator in func.decorator_list:
        ## @app.cell(...) is a call of app.cell; @app.cell is the attribute
        if isinstance(decorator, ast.Call):
            target = decorator.func
        else:
            target = decorator
        if isinstance(target, ast.Attribute) and target.attr == "cell":
            result = True
        else:
            pass
    return result


def no_comment_preserver(self, transform_func):
    """
    Replacement for `CommentPreserver.__call__`: return the transform
    unchanged (the text-editing transforms keep comments themselves).

    Args:
        - self (CommentPreserver): unused
        - transform_func (callable): a source transform

    Returns:
        - transform_func (callable): the same transform
    """
    return transform_func


def comment_texts(source):
    """
    Every comment in a source string, in order, as text.

    Args:
        - source (str): source code

    Returns:
        - comments (list of str): each comment's text
    """
    comments = []
    ## tokenize, so "#" inside strings isn't taken for a comment
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for token in tokens:
        if token.type == tokenize.COMMENT:
            comments.append(token.string.strip())
        else:
            pass
    return comments


def check_comments(notebook_text, marimo_text):
    """
    Comments of the notebook's code cells that are missing from the marimo
    version, or out of order. (Comments staying on the same line of code is
    guaranteed by apply_edits, which refuses edits that touch a comment.)

    Args:
        - notebook_text (str): the .ipynb file's contents
        - marimo_text (str): the converted .py

    Returns:
        - missing (list of str): comments not found, in order, in the output
          (empty if all are kept); cells that don't tokenize (e.g. `!ls`)
          are skipped
    """
    notebook = json.loads(notebook_text)
    expected = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ## a cell's source is a list of lines or one string
            source = cell["source"]
            if isinstance(source, list):
                source = "".join(source)
            else:
                pass
            ## magics/shell lines may not tokenize; skip those cells
            try:
                expected.extend(comment_texts(source))
            except (tokenize.TokenError, SyntaxError):
                pass
        else:
            pass

    ## match each expected comment to the next equal comment of the output;
    ## output-only comments (marimo's own) are skipped over
    found = comment_texts(marimo_text)
    position = 0
    missing = []
    for comment in expected:
        ## the next output comment with the same text, from position on
        match = None
        for index in range(position, len(found)):
            if match is None and found[index] == comment:
                match = index
            else:
                pass
        ## not there (or only earlier, i.e. out of order)
        if match is None:
            missing.append(comment)
        else:
            position = match + 1
    return missing


def convert(notebook_text):
    """
    Convert a notebook to marimo, keeping comments (see module docstring).

    Args:
        - notebook_text (str): the .ipynb file's contents

    Returns:
        - marimo_text (str): the marimo notebook (.py contents)
    """
    ## swap in the text-editing helpers only for the duration of the call
    patches = [
        unittest.mock.patch.object(to_ir, "_rename_in_scope", rename_in_scope_text),
        unittest.mock.patch.object(
            to_ir, "_transform_aug_assign", transform_aug_assign_text
        ),
        unittest.mock.patch.object(
            to_ir,
            "transform_remove_duplicate_imports",
            transform_remove_duplicate_imports_text,
        ),
        unittest.mock.patch.object(
            comment_preserver.CommentPreserver, "__call__", no_comment_preserver
        ),
    ]
    for patch in patches:
        patch.start()

    ## marimo skips a transform that raises, with only a logged warning
    ## (leaving cells marimo rejects); collect those warnings to fail on them
    failures = []
    handler = logging.Handler(level=logging.WARNING)
    handler.emit = lambda record: failures.append(record.getMessage())
    ## the logger marimo's transform pipeline writes to (named after to_ir)
    logger = logging.getLogger(to_ir.__name__)
    logger.addHandler(handler)

    ## marimo's own conversion (what `marimo convert` runs), then undo patches
    try:
        marimo_text = MarimoConvert.from_ipynb(notebook_text).to_py()
    finally:
        for patch in patches:
            patch.stop()
        logger.removeHandler(handler)

    ## a skipped transform means an incomplete conversion
    if len(failures) == 0:
        pass
    else:
        raise RuntimeError("marimo transforms failed:\n" + "\n".join(failures))
    return marimo_text


def main():
    """
    Command line: convert one notebook, refuse to write if comments are lost.

    Returns:
        - None (writes OUTPUT.py)
    """
    ## arguments: input notebook and output path
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("notebook", type=pathlib.Path, help="input .ipynb")
    parser.add_argument("output", type=pathlib.Path, help="output .py")
    ## parsed from sys.argv
    args = parser.parse_args()

    ## convert, end figure cells with plt.gca(), check every comment made it
    notebook_text = args.notebook.read_text()
    marimo_text = end_figure_cells(convert(notebook_text))
    missing = check_comments(notebook_text, marimo_text)

    ## write only a complete conversion
    if len(missing) == 0:
        args.output.write_text(marimo_text)
        print(f"wrote {args.output}")
    else:
        lost = "\n".join(f"  {text}" for text in missing)
        raise RuntimeError(f"{args.notebook}: comments lost or moved:\n{lost}")
    return None


if __name__ == "__main__":
    main()
