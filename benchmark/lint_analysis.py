"""Analysis lint, for CI: python3 benchmark/lint_analysis.py exits 1 on any violation.

1. No reported number takes a maximum over a learning curve: benchmark/analyze.py may not use a
   max-like reduction (max(), .max(), np.max/amax/nanmax, argmax, idxmax, nlargest, cummax,
   np.maximum/fmax, whose .accumulate is a running max), whether called, passed as a function,
   imported or named as a string ("max"), unless its line carries
   `# lint: allow-max <reason>`.
2. Every reported table names its checkpoint rule: analyze.py writes tables only in
   write_tables, which calls check_tables (the runtime check of that rule) before any write.
3. The train, val, test and lockbox level ids in common/rl-manager/utils/levels.py are pairwise
   disjoint, and EVAL_SPLITS holds exactly the named eval ranges.
"""
import ast
import itertools
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYZE = os.path.join(HERE, "analyze.py")
RL_MANAGER = os.path.join(os.path.dirname(HERE), "common", "rl-manager")

REDUCTIONS = {"max", "amax", "nanmax", "argmax", "nanargmax", "idxmax", "nlargest", "cummax"}
MAX_LIKE = REDUCTIONS | {"maximum", "fmax"}     # np.maximum.accumulate: a running max
PRAGMA = re.compile(r"#\s*lint:\s*allow-max\s+\S")
WRITERS = {"to_csv", "to_parquet", "to_json", "to_excel", "to_latex", "to_markdown", "to_html",
           "to_pickle", "to_feather"}


def max_like_violations(source: str, filename: str) -> list[str]:
    lines = source.splitlines()
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in MAX_LIKE:
            name = "." + node.attr
        elif isinstance(node, ast.Name) and node.id in MAX_LIKE:
            name = node.id
        elif isinstance(node, ast.Constant) and node.value in REDUCTIONS:     # .agg("max")
            name = repr(node.value)
        elif isinstance(node, ast.alias) and node.name.split(".")[-1] in MAX_LIKE:
            name = f"import {node.name}"
        else:
            continue
        if not any(PRAGMA.search(line) for line in lines[node.lineno - 1:node.end_lineno]):
            found.append(f"{filename}:{node.lineno}: max-like reduction {name} (a reported number "
                         f"may not maximise over a learning curve; `# lint: allow-max <reason>` "
                         f"if it does not)")
    return found


def table_write_violations(source: str, filename: str) -> list[str]:
    tree = ast.parse(source)
    writer = next((node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "write_tables"), None)
    if writer is None:
        return [f"{filename}: no write_tables(), the one place tables may be written"]
    inside = {id(node) for node in ast.walk(writer)}

    def calls(root, names):
        return [node for node in ast.walk(root) if isinstance(node, ast.Call) and (
            isinstance(node.func, ast.Attribute) and node.func.attr in names
            or isinstance(node.func, ast.Name) and node.func.id in names)]

    found = [f"{filename}:{node.lineno}: {ast.unparse(node.func)}() outside write_tables"
             for node in calls(tree, WRITERS) if id(node) not in inside]
    checks = [node.lineno for node in calls(writer, {"check_tables"})]
    writes = [node.lineno for node in calls(writer, WRITERS)]
    if not checks or (writes and min(writes) < min(checks)):
        found.append(f"{filename}:{writer.lineno}: write_tables must call check_tables before "
                     f"writing")
    return found


def split_violations() -> list[str]:
    if RL_MANAGER not in sys.path:
        sys.path.insert(0, RL_MANAGER)
    from utils import levels
    splits = {"train": levels.TRAIN_LEVELS, "val": levels.VAL_LEVELS,
              "test": levels.TEST_LEVELS, "lockbox": levels.LOCKBOX_LEVELS}
    found = []
    for (a, ids_a), (b, ids_b) in itertools.combinations(splits.items(), 2):
        shared = sorted(set(ids_a) & set(ids_b))
        if shared:
            found.append(f"levels.py: {a} and {b} share {len(shared)} level ids, e.g. {shared[:3]}")
    if levels.EVAL_SPLITS != {name: splits[name] for name in ("val", "test", "lockbox")}:
        found.append("levels.py: EVAL_SPLITS is not exactly the val, test and lockbox ranges")
    return found


def main() -> int:
    with open(ANALYZE) as f:
        source = f.read()
    found = (max_like_violations(source, ANALYZE) + table_write_violations(source, ANALYZE)
             + split_violations())
    for line in found:
        print(line)
    print(f"lint_analysis: {len(found)} violation(s)" if found else "lint_analysis: ok")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
