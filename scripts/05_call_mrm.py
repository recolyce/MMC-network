"""
Compute morphology-representative features (MRMs) from significant MPGs.

Keep the W rows of significant genes, aggregate them onto each morphology
dimension with gene weights, and optionally compute empirical P values and FDR
from permuted W matrices.
"""

# Gene-to-STRING identifier map, used to align genes with the rows of W.
STRING_MAPPING = "<STRING id mapping table for {prefix}>"
# MPG directory for this organ. Call tables are in its mpgs/ subdirectory.
MPG_DIR = "<MPG directory for {dataset}/{prefix}>"
# MRM directory for this organ. Score tables are written to its mrms/ subdirectory.
MRM_DIR = "<MRM directory for {dataset}/{prefix}>"
# MMC directory for this organ. W is at W_matrix/learned_W_matrix.npy.
W_DIR = "<MMC result directory for {dataset}/{prefix}>"

import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def load_gene_order(prefix):
    mapping_path = STRING_MAPPING.format(prefix=prefix)
    genes_df = pd.read_csv(mapping_path, sep="\t", index_col=0)
    genes_df = genes_df.drop_duplicates(subset=["stringId"], keep="first")
    gene_names = genes_df["queryItem"].tolist()
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    return gene_names, gene_to_idx


def load_significant_mpgs(mpg_path, q_value, top_k=None):
    mpg_df = pd.read_csv(mpg_path, sep="\t", index_col=0)

    if "is_significant" in mpg_df.columns:
        sig_df = mpg_df[mpg_df["is_significant"].astype(bool)].copy()
    elif "p_fdr" in mpg_df.columns:
        sig_df = mpg_df[mpg_df["p_fdr"] <= q_value].copy()
    else:
        raise ValueError(
            f"Cannot determine significant genes from {mpg_path}. "
            "Expected column `is_significant` or `p_fdr`."
        )

    if sig_df.empty:
        raise ValueError(f"No significant MPG genes found in {mpg_path}.")

    if top_k is not None and top_k > 0:
        sort_col = "z_score" if "z_score" in sig_df.columns else "ipi_raw"
        sig_df = sig_df.sort_values(sort_col, ascending=False).head(top_k).copy()

    return sig_df


def compute_gene_weights(sig_df, weight_mode):
    if weight_mode == "uniform":
        weights = np.ones(len(sig_df), dtype=np.float32)
    elif weight_mode == "zscore":
        if "z_score" not in sig_df.columns:
            raise ValueError("`z_score` column is required for weight_mode='zscore'.")
        weights = sig_df["z_score"].to_numpy(dtype=np.float32)
    elif weight_mode == "ipi":
        if "ipi_raw" not in sig_df.columns:
            raise ValueError("`ipi_raw` column is required for weight_mode='ipi'.")
        weights = sig_df["ipi_raw"].to_numpy(dtype=np.float32)
    else:
        raise ValueError(f"Unsupported weight_mode: {weight_mode}")

    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.clip(weights, a_min=0.0, a_max=None)
    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("All gene weights are zero after processing.")
    return weights / weight_sum


def aggregate_mrm_scores(w_sub, gene_weights, agg_mode):
    if agg_mode == "abs_sum":
        feature_scores = np.sum(np.abs(w_sub) * gene_weights[:, None], axis=0)
    elif agg_mode == "sq_sum":
        feature_scores = np.sum((w_sub ** 2) * gene_weights[:, None], axis=0)
    elif agg_mode == "positive_sum":
        feature_scores = np.sum(np.clip(w_sub, a_min=0.0, a_max=None) * gene_weights[:, None], axis=0)
    else:
        raise ValueError(f"Unsupported agg_mode: {agg_mode}")
    return feature_scores.astype(np.float32, copy=False)


def compute_mrm_scores_for_w(w_path, gene_indices, gene_weights, agg_mode):
    learned_w = np.load(w_path).astype(np.float32, copy=False)
    w_sub = learned_w[np.asarray(gene_indices)]
    return aggregate_mrm_scores(w_sub, gene_weights, agg_mode), learned_w.shape


def build_score_df(mrm_scores, weight_mode, agg_mode, n_significant_mpg):
    sorted_idx = np.argsort(mrm_scores)[::-1]
    score_df = pd.DataFrame(
        {
            "mrm_score": mrm_scores[sorted_idx],
            "feature_idx": sorted_idx,
        }
    )
    score_df["rank"] = np.arange(1, len(score_df) + 1)
    score_df["n_significant_mpg"] = n_significant_mpg
    score_df["weight_mode"] = weight_mode
    score_df["agg_mode"] = agg_mode
    return score_df, sorted_idx


def discover_permutation_w_paths(w_dir, n_perms=None):
    if n_perms is not None:
        if n_perms <= 0:
            raise ValueError("n_perms must be positive when provided.")
        paths = [
            os.path.join(w_dir, f"learned_W_matrix_perm_{p}.npy")
            for p in range(1, n_perms + 1)
        ]
        missing_paths = [path for path in paths if not os.path.exists(path)]
        if missing_paths:
            raise FileNotFoundError(
                "Missing permutation W matrix files, first missing file: "
                f"{missing_paths[0]}"
            )
        return paths

    paths = glob(os.path.join(w_dir, "learned_W_matrix_perm_*.npy"))
    if not paths:
        raise ValueError(f"No permutation W matrix files found in {w_dir}.")

    def perm_index(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return int(stem.rsplit("_", 1)[1])

    return sorted(paths, key=perm_index)


def empirical_p_values(true_scores, null_matrix, one_sided):
    n_perm = null_matrix.shape[0]
    if one_sided == "greater":
        extreme_count = (null_matrix >= true_scores[None, :]).sum(axis=0)
    elif one_sided == "less":
        extreme_count = (null_matrix <= true_scores[None, :]).sum(axis=0)
    else:
        raise ValueError("one_sided must be 'greater' or 'less'.")
    return (extreme_count) / (n_perm)


def main():
    parser = argparse.ArgumentParser(
        description="Compute MRM scores using only significant MPG genes."
    )
    parser.add_argument("--prefix", type=str, required=True, help="Cancer / sample prefix.")
    parser.add_argument("--dataset", type=str, default="hest1k")
    parser.add_argument(
        "--mpg_path",
        type=str,
        default=None,
        help="Optional explicit MPG result file path.",
    )
    parser.add_argument(
        "--w_path",
        type=str,
        default=None,
        help="Optional explicit learned W matrix path.",
    )
    parser.add_argument(
        "--q_value",
        type=float,
        default=0.05,
        help="Used only when MPG file does not contain `is_significant`.",
    )
    parser.add_argument(
        "--top_k_genes",
        type=int,
        default=None,
        help="Optionally keep only the top-k significant MPG genes.",
    )
    parser.add_argument(
        "--weight_mode",
        type=str,
        default="zscore",
        choices=["uniform", "zscore", "ipi"],
        help="How to weight significant MPG genes when aggregating morph features.",
    )
    parser.add_argument(
        "--agg_mode",
        type=str,
        default="abs_sum",
        choices=["abs_sum", "sq_sum", "positive_sum"],
        help="How to aggregate W[g, a] across significant genes.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Optional explicit output path.",
    )
    parser.add_argument(
        "--permutation",
        action="store_true",
        help="Use learned_W_matrix_perm_*.npy files to compute empirical MRM p/FDR.",
    )
    parser.add_argument(
        "--n_perms",
        type=int,
        default=None,
        help="Number of permutation W matrices to use. Defaults to all discovered files.",
    )
    parser.add_argument(
        "--tail",
        type=str,
        default="greater",
        choices=["greater", "less"],
        help="One-sided empirical test direction for MRM permutation.",
    )
    args = parser.parse_args()

    mpg_base_dir = MPG_DIR.format(dataset=args.dataset, prefix=args.prefix)
    mrm_base_dir = MRM_DIR.format(dataset=args.dataset, prefix=args.prefix)
    mmc_base_dir = W_DIR.format(dataset=args.dataset, prefix=args.prefix)
    mpg_path = args.mpg_path or os.path.join(
        mpg_base_dir,
        "mpgs",
        "mpgs_permutation_quantile_threshold_q1_gene_score.tsv",
    )
    w_path = args.w_path or os.path.join(mmc_base_dir, "W_matrix", "learned_W_matrix.npy")
    w_dir = os.path.dirname(w_path)

    if not os.path.exists(mpg_path):
        raise FileNotFoundError(f"MPG file not found: {mpg_path}")
    if not os.path.exists(w_path):
        raise FileNotFoundError(f"W matrix file not found: {w_path}")

    _, gene_to_idx = load_gene_order(args.prefix)
    sig_df = load_significant_mpgs(mpg_path, q_value=args.q_value, top_k=args.top_k_genes)

    gene_indices = []
    kept_gene_names = []
    for gene in sig_df.index.tolist():
        if gene in gene_to_idx:
            gene_indices.append(gene_to_idx[gene])
            kept_gene_names.append(gene)

    if len(gene_indices) == 0:
        raise ValueError("None of the significant MPG genes matched the W matrix gene order.")

    if len(gene_indices) != len(sig_df):
        sig_df = sig_df.loc[kept_gene_names].copy()

    gene_weights = compute_gene_weights(sig_df, args.weight_mode)
    mrm_scores, w_shape = compute_mrm_scores_for_w(
        w_path,
        gene_indices,
        gene_weights,
        args.agg_mode,
    )
    score_df, sorted_idx = build_score_df(
        mrm_scores,
        args.weight_mode,
        args.agg_mode,
        len(sig_df),
    )

    if args.permutation:
        perm_w_paths = discover_permutation_w_paths(w_dir, n_perms=args.n_perms)
        null_scores = []
        for perm_w_path in perm_w_paths:
            perm_scores, perm_w_shape = compute_mrm_scores_for_w(
                perm_w_path,
                gene_indices,
                gene_weights,
                args.agg_mode,
            )
            if perm_w_shape != w_shape:
                raise ValueError(
                    f"Permutation W shape mismatch for {perm_w_path}: "
                    f"expected {w_shape}, got {perm_w_shape}."
                )
            null_scores.append(perm_scores)

        null_matrix = np.stack(null_scores, axis=0)
        null_std_ddof = 1 if null_matrix.shape[0] > 1 else 0
        p_empirical = empirical_p_values(mrm_scores, null_matrix, args.tail)
        reject, p_fdr, _, _ = multipletests(
            p_empirical,
            alpha=args.q_value,
            method="fdr_bh",
        )
        null_mean = null_matrix.mean(axis=0)
        null_std = null_matrix.std(axis=0, ddof=null_std_ddof) + 1e-12
        z_score = (mrm_scores - null_mean) / null_std

        score_df["null_mean"] = null_mean[sorted_idx]
        score_df["null_std"] = null_std[sorted_idx]
        score_df["z_score"] = z_score[sorted_idx]
        score_df["p_empirical"] = p_empirical[sorted_idx]
        score_df["p_fdr"] = p_fdr[sorted_idx]
        score_df["is_significant"] = reject[sorted_idx]
        score_df["n_permutation"] = null_matrix.shape[0]
        score_df["tail"] = args.tail

    if args.out_path is None:
        out_dir = os.path.join(mrm_base_dir, "mrms")
        os.makedirs(out_dir, exist_ok=True)
        topk_suffix = f"_top{args.top_k_genes}" if args.top_k_genes else ""
        permutation_suffix = (
            f"_permutation_empirical_pvalue_fdr_q{args.q_value}"
            if args.permutation
            else ""
        )
        out_path = os.path.join(
            out_dir,
            f"mrm_scores_from_mpg_weight-{args.weight_mode}_agg-{args.agg_mode}"
            f"{topk_suffix}{permutation_suffix}.tsv",
        )
    else:
        out_path = args.out_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    score_df.to_csv(out_path, sep="\t", index=False)

    selected_gene_path = out_path.replace(".tsv", ".selected_mpgs.tsv")
    selected_gene_df = sig_df.copy()
    selected_gene_df["gene_weight"] = gene_weights
    # selected_gene_df.to_csv(selected_gene_path, sep="\t")

    print("-" * 30)
    print(f"Prefix: {args.prefix}")
    print(f"Significant MPG used: {len(sig_df)}")
    print(f"W shape: {w_shape}")
    print(f"Weight mode: {args.weight_mode}")
    print(f"Aggregation mode: {args.agg_mode}")
    if args.permutation:
        print(f"Permutation rounds used: {len(perm_w_paths)}")
        print(f"FDR threshold (q): {args.q_value}")
        print(f"Significant MRM features (FDR): {int(reject.sum())}")
    print(f"Saved MRM scores: {out_path}")
    print(f"Saved selected MPG table: {selected_gene_path}")
    print("-" * 30)


if __name__ == "__main__":
    main()
