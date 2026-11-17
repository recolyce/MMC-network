"""
Score genes with the learned W.

For the true round and each permuted W, write an ipi_round table of gene
predictability scores. These tables are the input to MPG calling.
"""

# HEST UNI features, one h5 file per sample.
HEST_UNI_H5_DIR = "<HEST UNI features, one h5 file per sample>"
# HEST expression matrices for this organ, one h5 file per sample.
HEST_COUNT_GLOB = "<HEST processed counts for {prefix}>/*.h5"
# Output directory for ipi_round_*.tsv gene-score tables.
GENE_SCORE_DIR = "<gene score directory for {dataset}/{prefix}>"
# Directory of the learned W and bias.
W_DIR = "<learned W directory for {dataset}/{prefix}>"
# Morphology prior adjacency matrix.
MORPHOLOGY_ADJACENCY = "<morphology prior adjacency A_M.npy for {prefix}>"
# Gene-to-STRING identifier map for this organ.
STRING_MAPPING = "<STRING id mapping table for {prefix}>"
# PPI graph for this organ.
PPI_GRAPH = "<PPI graph pickle for {prefix}>"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03-MMC_network"))

from data import MMCData
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import compute_gene_predictability_score, load_al_from_nx


def w_paths(w_dir, round_idx):
    if round_idx == 0:
        return (
            os.path.join(w_dir, "learned_W_matrix.npy"),
            os.path.join(w_dir, "learned_W_bias_matrix.npy"),
        )
    return (
        os.path.join(w_dir, f"learned_W_matrix_perm_{round_idx}.npy"),
        os.path.join(w_dir, f"learned_W_bias_matrix_perm_{round_idx}.npy"),
    )


def score_results_to_df(score_results, gene_names):
    gene_score = np.asarray(score_results["gene_score"])
    sorted_idx = score_results.get("gene_sorted_idx", np.argsort(gene_score)[::-1])

    result_df = pd.DataFrame({"gene": gene_names})
    for key, value in score_results.items():
        if key == "gene_sorted_idx":
            continue
        value = np.asarray(value)
        if value.ndim == 1 and len(value) == len(gene_names):
            result_df[key] = value

    # Keep the historical column name so downstream FDR scripts remain compatible.
    result_df["ipi_scores"] = result_df["gene_score"]
    result_df = result_df.iloc[sorted_idx].copy()
    result_df["rank"] = np.arange(1, len(result_df) + 1)
    return result_df.set_index("gene")


def main():
    parser = argparse.ArgumentParser(
        description="Compute full per-gene MMC score result tables from saved W/bias matrices."
    )
    parser.add_argument("--prefix", type=str, default="Kidney", help="Dataset prefix")
    parser.add_argument("--dataset", type=str, default="hest1k")
    parser.add_argument("--n_perms", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--nsamples",
        type=int,
        default=None,
        help="Use at most this many samples when building MMCData for quick scoring runs.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip rounds whose score table already exists.",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip rounds whose W or bias file is missing instead of failing.",
    )
    parser.add_argument(
        "--center_X",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Center X_gene and X_morph before scoring.",
    )
    parser.add_argument(
        "--scale_X",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Scale X_gene and X_morph before scoring.",
    )
    args = parser.parse_args()

    np.random.default_rng(args.seed)

    prefix = args.prefix
    dataset = args.dataset
    input_feat = HEST_UNI_H5_DIR
    input_omic_pattern = HEST_COUNT_GLOB.format(prefix=prefix)

    score_dir = GENE_SCORE_DIR.format(dataset=dataset, prefix=prefix)
    w_dir = W_DIR.format(dataset=dataset, prefix=prefix)
    os.makedirs(score_dir, exist_ok=True)
    a_morpho_np = np.load(MORPHOLOGY_ADJACENCY.format(prefix=prefix))
    a_morpho_np = np.clip(a_morpho_np, 0, 1)
    l_morpho_np = np.diag(np.sum(a_morpho_np, axis=1)) - a_morpho_np

    genes_df = pd.read_csv(
        STRING_MAPPING.format(prefix=prefix),
        sep="\t",
        index_col=0,
    )
    genes_df.drop_duplicates(subset=["stringId"], keep="first", inplace=True)
    gene_names = genes_df["queryItem"].to_list()

    with open(PPI_GRAPH.format(prefix=prefix), "rb") as file:
        g_molecule = pickle.load(file)
    _, _, _, l_molecule_np = load_al_from_nx(g_molecule, genes_df["stringId"].tolist())
    # _, l_molecule_np, _, _ = load_al_from_nx(g_molecule, genes_df["stringId"].tolist())

    mmc_dataset = MMCData(
        feat_dir=input_feat,
        omic_pth_pattern=input_omic_pattern,
        gene_list=gene_names,
        preprocessing=True,
        dataset=dataset,
        num_sample=args.nsamples,
    )
    print(f"Length of scoring data: {len(mmc_dataset)}")
    full_morph_x_mat = np.asarray(mmc_dataset.morph_feats, dtype=np.float32)
    full_expr_x_mat = np.asarray(mmc_dataset.exprs, dtype=np.float32)

    for p in tqdm(range(0, args.n_perms + 1)):
        score_out = os.path.join(score_dir, f"ipi_round_{p}.tsv")
        if args.skip_existing and os.path.exists(score_out):
            continue

        w_mat_path, w_bias_path = w_paths(w_dir, p)
        if not os.path.exists(w_mat_path) or not os.path.exists(w_bias_path):
            if args.skip_missing:
                print(f"Skipping round {p}: missing W or bias.")
                continue
            raise FileNotFoundError(f"Missing W/bias for round {p}: {w_mat_path}, {w_bias_path}")

        learned_w = np.load(w_mat_path)
        learned_bias = np.load(w_bias_path)

        score_results = compute_gene_predictability_score(
            X_gene=full_expr_x_mat,
            X_morph=full_morph_x_mat,
            W=learned_w,
            bias=learned_bias,
            L_gene=l_molecule_np,
            L_morph=l_morpho_np,
            W_orientation="gene_by_morph",
            center_X=args.center_X,
            scale_X=args.scale_X,
        )
        result_df = score_results_to_df(score_results, gene_names)
        result_df.to_csv(score_out, sep="\t")


if __name__ == "__main__":
    main()
