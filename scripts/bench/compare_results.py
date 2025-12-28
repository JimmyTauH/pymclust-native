#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Compare JSONL outputs from R mclust and pymclust-native

def _safe_arr(x):
    if x is None:
        return None
    return np.asarray(x)


def load_jsonl(path):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def join_by_key(rows_a, rows_b, key=("dataset_id","model")):
    idx_b = {(r[key[0]], r[key[1]]): r for r in rows_b}
    pairs = []
    for r in rows_a:
        k = (r[key[0]], r[key[1]])
        if k in idx_b:
            pairs.append((r, idx_b[k]))
    return pairs


def _maybe_ari_nmi(r_rec, p_rec):
    # If labels or posterior exist in both, compute ARI/NMI; otherwise return (None, None)
    def extract_labels(rec):
        if rec.get("labels") is not None:
            return np.asarray(rec["labels"]).ravel()
        if rec.get("posterior") is not None:
            P = np.asarray(rec["posterior"])
            if P.ndim == 2 and P.shape[1] > 0:
                return P.argmax(axis=1)
        return None
    y_r = extract_labels(r_rec)
    y_p = extract_labels(p_rec)
    if y_r is None or y_p is None or y_r.shape[0] != y_p.shape[0]:
        return None, None
    return float(adjusted_rand_score(y_r, y_p)), float(normalized_mutual_info_score(y_r, y_p))


def compare_metrics(r_rec, p_rec):
    out = {
        "dataset_id": r_rec["dataset_id"],
        "model": r_rec["model"],
        "bic_diff": None,
        "icl_diff": None,
        "loglik_diff": None,
        "ari": None,
        "nmi": None,
        "error": None,
    }
    if "error" in r_rec or "error" in p_rec:
        out["error"] = r_rec.get("error") or p_rec.get("error")
        return out
    # Basic metrics
    for k, field in [("loglik_diff", "loglik"), ("bic_diff", "bic"), ("icl_diff", "icl")]:
        rv = r_rec.get(field)
        pv = p_rec.get(field)
        if rv is not None and pv is not None:
            out[k] = float(pv - rv)
    # Optional clustering metrics
    ari, nmi = _maybe_ari_nmi(r_rec, p_rec)
    if ari is not None:
        out["ari"] = ari
        out["nmi"] = nmi
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r", dest="r_path", type=str, default="pymclust-native/.bench_results/mclust_results.jsonl")
    p.add_argument("--py", dest="py_path", type=str, default="pymclust-native/.bench_results/pymclust_results.jsonl")
    p.add_argument("--out", type=str, default="pymclust-native/.bench_results/compare_summary.jsonl")
    args = p.parse_args()

    r_rows = load_jsonl(args.r_path)
    py_rows = load_jsonl(args.py_path)
    pairs = join_by_key(r_rows, py_rows)

    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.out, 'w') as f:
        for r_rec, p_rec in pairs:
            cmp = compare_metrics(r_rec, p_rec)
            f.write(json.dumps(cmp) + "\n")

    print(f"Compared {len(pairs)} pairs; wrote {args.out}")

if __name__ == "__main__":
    main()
