#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np

from pymclust_native.api import fit_mclust
from pymclust_native.models import ModelName

# Run pymclust-native over reference datasets and dump results in JSON lines

MODEL_LIST = [
    # Core diagonal/spherical for sanity
    "EII", "VII", "EEI", "VVI", "VEI", "EVI",
    # New families
    "VEE", "EVE", "VVE", "EVV", "EEV", "VEV",
]


def _safe_scalar_first(x, default=None):
    arr = np.asarray(x)
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])


def to_serializable(result, model: str, dataset_id: str):
    out = {
        "dataset_id": dataset_id,
        "model": model,
        "loglik": float(result.loglik),
        "bic": float(result.bic) if getattr(result, "bic", None) is not None else None,
        "icl": float(getattr(result, "icl", None)) if getattr(result, "icl", None) is not None else None,
        "weights": np.asarray(result.weights).tolist(),
        "means": np.asarray(result.means).tolist(),
    }
    fam = model
    cov = result.covariances

    try:
        if fam == "VVV":
            out["cov"] = {"type": fam, "Sigmas": np.asarray(cov).tolist()}
        elif fam == "EEE":
            out["cov"] = {"type": fam, "Sigma": np.asarray(cov).tolist()}
        elif fam in ("EII", "VII"):
            lam = np.asarray(cov).ravel()
            out["cov"] = {"type": fam, "lambda": lam.tolist()}
        elif fam == "EEI":
            lam, A = cov
            lam_val = _safe_scalar_first(lam, default=None)
            out["cov"] = {"type": fam, "lambda": lam_val, "A": np.asarray(A).reshape(-1).tolist()}
        elif fam == "VVI":
            out["cov"] = {"type": fam, "diag": np.asarray(cov).tolist()}
        elif fam == "VEI":
            lam_g, A = cov
            out["cov"] = {"type": fam, "lambda_g": np.asarray(lam_g).ravel().tolist(), "A": np.asarray(A).reshape(-1).tolist()}
        elif fam == "EVI":
            lam, A_g = cov
            lam_val = _safe_scalar_first(lam, default=None)
            out["cov"] = {"type": fam, "lambda": lam_val, "A_g": np.asarray(A_g).tolist()}
        elif fam in ("VEE", "EVE", "VVE", "EVV", "EEV", "VEV"):
            if fam == "VEE":
                lam_g, A, D = cov
                out["cov"] = {
                    "type": fam,
                    "lambda_g": np.asarray(lam_g).ravel().tolist(),
                    "A": np.asarray(A).reshape(-1).tolist(),
                    "D": np.asarray(D).tolist(),
                }
            elif fam == "EVE":
                lam, A_g, D = cov
                out["cov"] = {
                    "type": fam,
                    "lambda": _safe_scalar_first(lam, default=None),
                    "A_g": np.asarray(A_g).tolist(),
                    "D": np.asarray(D).tolist(),
                }
            elif fam == "VVE":
                lam_g, A_g, D = cov
                out["cov"] = {
                    "type": fam,
                    "lambda_g": np.asarray(lam_g).ravel().tolist(),
                    "A_g": np.asarray(A_g).tolist(),
                    "D": np.asarray(D).tolist(),
                }
            elif fam == "EVV":
                lam, A_g, D_g = cov
                out["cov"] = {
                    "type": fam,
                    "lambda": _safe_scalar_first(lam, default=None),
                    "A_g": np.asarray(A_g).tolist(),
                    "D_g": np.asarray(D_g).tolist(),
                }
            elif fam == "EEV":
                lam, A, D_g = cov
                out["cov"] = {
                    "type": fam,
                    "lambda": _safe_scalar_first(lam, default=None),
                    "A": np.asarray(A).reshape(-1).tolist(),
                    "D_g": np.asarray(D_g).tolist(),
                }
            else:  # VEV
                lam_g, A, D_g = cov
                out["cov"] = {
                    "type": fam,
                    "lambda_g": np.asarray(lam_g).ravel().tolist(),
                    "A": np.asarray(A).reshape(-1).tolist(),
                    "D_g": np.asarray(D_g).tolist(),
                }
        else:
            out["cov"] = {"type": "FULL", "Sigmas": np.asarray(cov).tolist()}
    except Exception as e:
        out["cov"] = {"type": fam, "error": f"serialization_failed: {e}", "raw_repr": str(type(cov))}
    return out


def run_one(dataset_path: Path, models, var_floor: float, full_jitter: float, restarts: int):
    data = np.load(dataset_path)
    X = data["X"]
    outputs = []
    for model in models:
        try:
            model_enum = getattr(ModelName, model)
            res = fit_mclust(
                X,
                G_list=[2, 3],  # small set for runtime; adjust as needed
                models=[model_enum],
                n_init=restarts,
                selection_criterion="BIC",
                var_floor=var_floor,
                full_jitter=full_jitter,
            )
            outputs.append(to_serializable(res, model, dataset_path.stem))
        except Exception as e:
            outputs.append({
                "dataset_id": dataset_path.stem,
                "model": model,
                "error": str(e),
            })
    return outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, default="pymclust-native/.bench_data/manifest.jsonl")
    p.add_argument("--out", type=str, default="pymclust-native/.bench_results/pymclust_results.jsonl")
    p.add_argument("--models", type=str, nargs="*", default=MODEL_LIST)
    p.add_argument("--var-floor", type=float, default=1e-6)
    p.add_argument("--full-jitter", type=float, default=1e-9)
    p.add_argument("--restarts", type=int, default=3)
    args = p.parse_args()

    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)

    outputs = []
    with open(args.manifest, "r") as f:
        for line in f:
            rec = json.loads(line)
            ds_path = Path(rec["path"])
            outputs.extend(run_one(ds_path, args.models, args.var_floor, args.full_jitter, args.restarts))

    with open(args.out, "w") as f:
        for rec in outputs:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(outputs)} results to {args.out}")


if __name__ == "__main__":
    main()
