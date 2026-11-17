"""
Call morphology-predictable genes (MPGs).

Compare the true-round gene score with the global quantile of each permutation
round. A gene is an MPG when its score exceeds the mean of those quantiles.
"""

# Directory of gene-score tables, including ipi_round_0.tsv and the permutations.
GENE_SCORE_DIR = "<gene score directory for {dataset}/{prefix}>"
# Output directory for the MPG call tables.
MPG_DIR = "<MPG output directory for {dataset}/{prefix}>"

import argparse
import os
from glob import glob

import numpy as np
import pandas as pd


def discover_permutation_paths(base_dir, n_perms=None):
    if n_perms is not None:
        if n_perms <= 0:
            raise ValueError("n_perms must be positive when provided.")
        paths = [
            os.path.join(base_dir, f"ipi_round_{idx}.tsv")
            for idx in range(1, n_perms + 1)
        ]
        missing = [path for path in paths if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(f"Missing permutation score file: {missing[0]}")
        return paths

    paths = glob(os.path.join(base_dir, "ipi_round_*.tsv"))
    paths = [path for path in paths if not path.endswith("ipi_round_0.tsv")]

    def round_idx(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return int(stem.rsplit("_", 1)[1])

    return sorted(paths, key=round_idx)


def load_true_scores(base_dir, col_name):
    true_path = os.path.join(base_dir, "ipi_round_0.tsv")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"True score file not found: {true_path}")

    true_df = pd.read_csv(true_path, sep="\t", index_col=0)
    if col_name not in true_df.columns:
        raise ValueError(
            f"Column `{col_name}` not found in {true_path}. "
            f"Available columns: {list(true_df.columns)}"
        )
    return true_df.index.to_list(), true_df[col_name].to_numpy(dtype=np.float64)


def load_permutation_quantiles(base_dir, col_name, quantile=0.95, n_perms=None):
    perm_paths = discover_permutation_paths(base_dir, n_perms=n_perms)
    if not perm_paths:
        raise ValueError(f"No permutation score files found in {base_dir}.")

    quantiles = []
    for path in perm_paths:
        perm_df = pd.read_csv(path, sep="\t", index_col=0)
        if col_name not in perm_df.columns:
            raise ValueError(
                f"Column `{col_name}` not found in {path}. "
                f"Available columns: {list(perm_df.columns)}"
            )
        scores = perm_df[col_name].to_numpy(dtype=np.float64)
        scores = scores[np.isfinite(scores)]
        if len(scores) == 0:
            raise ValueError(f"No finite scores found in {path}.")
        quantiles.append(float(np.quantile(scores, quantile)))

    return np.asarray(quantiles, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Call significant genes by comparing round 0 scores against the mean "
            "of per-permutation global quantile thresholds."
        )
    )
    parser.add_argument("--prefix", type=str, default="Breast", help="Dataset prefix")
    parser.add_argument("--dataset", type=str, default="hest1k")
    parser.add_argument("--n_perms", type=int, default=100, help="Number of permutation rounds")
    parser.add_argument(
        "--col_name",
        type=str,
        default="gene_score",
        help="Score column used in true and permutation score tables.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=1.0,
        help="Per-permutation global quantile used as the null threshold.",
    )
    args = parser.parse_args()

    if not (0.0 <= args.quantile <= 1.0):
        raise ValueError("--quantile must be between 0 and 1.")

    base_dir = GENE_SCORE_DIR.format(dataset=args.dataset, prefix=args.prefix)
    gene_list, true_scores = load_true_scores(base_dir, args.col_name)
    perm_quantiles = load_permutation_quantiles(
        base_dir,
        args.col_name,
        quantile=args.quantile,
        n_perms=args.n_perms,
    )
    threshold = float(np.mean(perm_quantiles))
    is_significant = true_scores > threshold

    results_df = pd.DataFrame(
        {
            "gene": gene_list,
            "ipi_raw": true_scores,
            "threshold": threshold,
            "is_significant": is_significant,
            "score_col": args.col_name,
            "quantile": args.quantile,
            "n_permutation": len(perm_quantiles),
            "permutation_quantile_mean": threshold,
            "permutation_quantile_std": float(np.std(perm_quantiles, ddof=1))
            if len(perm_quantiles) > 1
            else 0.0,
        }
    ).set_index("gene")
    results_df = results_df.sort_values("ipi_raw", ascending=False)
    significant_df = results_df[results_df["is_significant"]].copy()

    sig_count = int(results_df["is_significant"].sum())
    print("-" * 30)
    print(f"Permutation rounds used: {len(perm_quantiles)}")
    print(f"Score column: {args.col_name}")
    print(f"Permutation quantile: {args.quantile}")
    print(f"Mean quantile threshold: {threshold:.6g}")
    print(f"Significant genes: {sig_count}")
    print("-" * 30)

    out_dir = MPG_DIR.format(dataset=args.dataset, prefix=args.prefix)
    os.makedirs(out_dir, exist_ok=True)
    col_suffix = args.col_name.replace("/", "_")
    q_suffix = f"q{args.quantile:g}".replace(".", "p")
    out_path = os.path.join(
        out_dir,
        f"mpgs_permutation_quantile_threshold_{q_suffix}_{col_suffix}.tsv",
    )
    sig_out_path = os.path.join(
        out_dir,
        f"mpgs_permutation_quantile_threshold_{q_suffix}_{col_suffix}.significant.tsv",
    )
    results_df.to_csv(out_path, sep="\t")
    significant_df.to_csv(sig_out_path, sep="\t")
    print(f"Saved full results: {out_path}")
    print(f"Saved significant MPGs: {sig_out_path}")


if __name__ == "__main__":
    main()
