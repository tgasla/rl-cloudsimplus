"""benchmark/lint_analysis.py, the CI lint over the analysis.
Run: python3 -m pytest benchmark/tests/test_lint_analysis.py"""

import os
import shutil
import subprocess
import sys

import pytest

BENCHMARK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCHMARK)

import lint_analysis  # noqa: E402

with open(lint_analysis.ANALYZE) as _f:
    REAL = _f.read()

PLANTED = [
    "best = curve.max()",
    "best = curve.idxmax()",
    "best = curve.nlargest(1)",
    "best = max(curve)",
    "best = np.max(curve)",
    "best = np.amax(curve)",
    "best = curve.cummax()",
    "best = np.nanmax(curve)",
    "best = curve[np.argmax(curve)]",
    "best = curve.agg('max')",
    "best = curve.groupby(level=0).transform(max)",
    "best = getattr(curve, 'max')()",
    "from numpy import amax",
    "best = (curve\n        .rolling(10).mean()\n        .max())",
    "best = np.maximum.accumulate(curve)[-1]",
    "best = np.fmax.reduce(curve)",
]


def _plant(code: str) -> str:
    return REAL + "\n\ndef planted(curve):\n    " + code.replace("\n", "\n    ") + "\n"


def test_the_real_analysis_passes():
    assert lint_analysis.max_like_violations(REAL, "analyze.py") == []
    assert lint_analysis.table_write_violations(REAL, "analyze.py") == []
    assert lint_analysis.split_violations() == []


@pytest.mark.parametrize("code", PLANTED)
def test_each_planted_max_like_reduction_fails(code):
    found = lint_analysis.max_like_violations(_plant(code), "analyze.py")
    assert found and all("max-like reduction" in line for line in found)
    assert found[0].startswith(f"analyze.py:{len(REAL.splitlines()) + 4}:")


def test_the_pragma_needs_a_reason_and_its_own_line():
    allowed = "def f(c):\n    return c.max()  # lint: allow-max peak memory, not a score\n"
    assert lint_analysis.max_like_violations(allowed, "a.py") == []
    spanning = "def f(c):\n    return (c\n            .max())  # lint: allow-max not a score\n"
    assert lint_analysis.max_like_violations(spanning, "a.py") == []
    for refused in ("def f(c):\n    return c.max()  # lint: allow-max\n",
                    "def f(c):\n    # lint: allow-max not a score\n    return c.max()\n"):
        assert lint_analysis.max_like_violations(refused, "a.py")


def test_prose_about_maxima_is_not_a_reduction():
    prose = '"""Never report the max of a curve, nor .max() of it."""\n# max(curve) is out\n' \
            'LABEL = "maximum"\nmaxima = 0\n'
    assert lint_analysis.max_like_violations(prose, "a.py") == []


def test_tables_are_written_only_by_write_tables_after_check_tables():
    outside = REAL + "\n\ndef dump(table):\n    table.to_csv('x.csv')\n    to_parquet(table)\n"
    assert [line.split(": ", 1)[1] for line in
            lint_analysis.table_write_violations(outside, "a.py")] == [
        "table.to_csv() outside write_tables", "to_parquet() outside write_tables"]
    unchecked = REAL.replace("    check_tables(tables)\n    os.makedirs", "    os.makedirs")
    assert unchecked != REAL
    assert "must call check_tables" in lint_analysis.table_write_violations(unchecked, "a.py")[0]
    late = REAL.replace("    check_tables(tables)\n    os.makedirs", "    os.makedirs").replace(
        '        print(f"wrote {path}',
        '        check_tables(tables)\n        print(f"wrote {path}')
    assert "must call check_tables" in lint_analysis.table_write_violations(late, "a.py")[0]
    assert "no write_tables()" in lint_analysis.table_write_violations("x = 1\n", "a.py")[0]


def test_overlapping_splits_fail(monkeypatch):
    from utils import levels
    monkeypatch.setattr(levels, "VAL_LEVELS", range(99990, 100014))
    found = lint_analysis.split_violations()
    assert found[0] == "levels.py: train and val share 10 level ids, e.g. [99990, 99991, 99992]"
    assert found[1].startswith("levels.py: EVAL_SPLITS is not exactly")


def test_the_script_exits_non_zero_on_a_violation(tmp_path):
    """The CI invocation, on a copy of benchmark/ whose analyze.py has a planted .max()."""
    copy = tmp_path / "benchmark"
    copy.mkdir()
    shutil.copy(lint_analysis.__file__, copy)
    os.symlink(os.path.join(os.path.dirname(BENCHMARK), "common"), tmp_path / "common")
    (copy / "analyze.py").write_text(REAL)
    clean = subprocess.run([sys.executable, str(copy / "lint_analysis.py")], capture_output=True,
                           text=True)
    assert clean.returncode == 0 and "lint_analysis: ok" in clean.stdout, clean.stderr
    (copy / "analyze.py").write_text(_plant("best = curve.max()"))
    dirty = subprocess.run([sys.executable, str(copy / "lint_analysis.py")], capture_output=True,
                           text=True)
    assert dirty.returncode == 1 and "1 violation(s)" in dirty.stdout, dirty.stderr
