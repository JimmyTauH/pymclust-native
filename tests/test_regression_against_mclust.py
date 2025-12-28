import json
import os
import pathlib
import pytest

# Slow regression test scaffold: reads precomputed results and asserts coarse tolerances.
# Marked slow and mclust; will be skipped if result files are missing.

RESULTS_DIR = pathlib.Path("pymclust-native/.bench_results")
PY_PATH = RESULTS_DIR / "pymclust_results.jsonl"
R_PATH = RESULTS_DIR / "mclust_results.jsonl"

SLOW = pytest.mark.slow
MCLUST = pytest.mark.mclust


def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def join_pairs(py_rows, r_rows):
    idx = {(r["dataset_id"], r["model"]): r for r in r_rows}
    for p in py_rows:
        key = (p["dataset_id"], p["model"])
        if key in idx:
            yield p, idx[key]


@SLOW
@MCLUST
@pytest.mark.skipif(not (PY_PATH.exists() and R_PATH.exists()), reason="Benchmark result files not found; run bench scripts first.")
def test_bic_icl_close_to_mclust():
    py_rows = list(load_jsonl(PY_PATH))
    r_rows = list(load_jsonl(R_PATH))

    count = 0
    for p, r in join_pairs(py_rows, r_rows):
        if "error" in p or "error" in r:
            continue
        # tolerances
        def within(x, y):
            if x is None or y is None:
                return True
            thr = max(0.1 * abs(y), 5.0)
            return abs(x - y) <= thr
        assert within(p.get("bic"), r.get("bic"))
        assert within(p.get("icl"), r.get("icl"))
        count += 1

    assert count > 0, "No comparable pairs found in results."
