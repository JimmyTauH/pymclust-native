#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import numpy as np

# Minimal, reproducible reference datasets for benchmarking against R mclust
# Produces a small set by default; can be scaled via CLI options

def make_dataset(seed: int, n: int, d: int, G: int):
    rng = np.random.default_rng(seed)
    # Generate mixture means spread on a sphere
    means = rng.normal(size=(G, d))
    means = means / np.linalg.norm(means, axis=1, keepdims=True)
    means *= 3.0
    # Random orthogonal bases per component
    Ds = []
    for g in range(G):
        A = rng.normal(size=(d, d))
        Q, _ = np.linalg.qr(A)
        Ds.append(Q)
    Ds = np.stack(Ds, axis=0)
    # Diagonal scales (anisotropic)
    As = rng.lognormal(mean=0.0, sigma=0.5, size=(G, d))
    As = As / (np.prod(As, axis=1, keepdims=True) ** (1.0 / d))  # det(A)=1 normalization
    # Global scales per component
    lambdas = rng.lognormal(mean=0.0, sigma=0.25, size=(G,))
    # Build covariances
    covs = np.empty((G, d, d))
    for g in range(G):
        D = Ds[g]
        A = np.diag(As[g])
        covs[g] = lambdas[g] * D @ A @ D.T
    # Mixing weights
    weights = rng.dirichlet(alpha=np.ones(G))
    # Sample
    zs = rng.choice(G, size=n, p=weights)
    X = np.empty((n, d))
    for g in range(G):
        idx = np.where(zs == g)[0]
        if idx.size:
            X[idx] = rng.multivariate_normal(mean=means[g], cov=covs[g], size=idx.size)
    return {
        "X": X,
        "true_labels": zs,
        "means": means,
        "covs": covs,
        "weights": weights,
        "seed": seed,
        "n": n,
        "d": d,
        "G": G,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="pymclust-native/.bench_data", help="Output directory")
    p.add_argument("--seeds", type=int, nargs="*", default=[1, 2])
    p.add_argument("--sizes", type=int, nargs="*", default=[300])
    p.add_argument("--dims", type=int, nargs="*", default=[2, 3])
    p.add_argument("--components", type=int, nargs="*", default=[2, 3])
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = []
    ds_id = 0
    for seed in args.seeds:
        for n in args.sizes:
            for d in args.dims:
                for G in args.components:
                    ds = make_dataset(seed=seed, n=n, d=d, G=G)
                    fname = outdir / f"ds_{ds_id:04d}.npz"
                    np.savez_compressed(
                        fname,
                        X=ds["X"],
                        true_labels=ds["true_labels"],
                        means=ds["means"],
                        covs=ds["covs"],
                        weights=ds["weights"],
                        seed=ds["seed"],
                        n=ds["n"],
                        d=ds["d"],
                        G=ds["G"],
                    )
                    manifest.append({
                        "dataset_id": f"ds_{ds_id:04d}",
                        "path": str(fname),
                        "seed": seed,
                        "n": n,
                        "d": d,
                        "G": G,
                    })
                    ds_id += 1

    with open(outdir / "manifest.jsonl", "w") as f:
        for rec in manifest:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(manifest)} datasets to {outdir}")


if __name__ == "__main__":
    main()
