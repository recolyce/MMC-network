"""
Define MMC modules.

Join the PPI of significant MPGs, the morphology adjacency of significant MRMs,
and the W between them into one graph. Cluster it spectrally, then assign each
module a unique Hallmark pathway name.
"""

import argparse
import os
import pickle
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding


# Repository root. PPI graphs, learned W, MPG tables, MRM tables, and the morphology prior live under it.
PROJECT_ROOT = Path("<repository root>")
# HEST-1k root. The gene identifier maps are in processed_data/string_mapping/.
HEST_ROOT = Path("<HEST-1k root>")


def normalized_adjacency_from_nx(graph, node_list):
    adjacency = nx.to_numpy_array(
        graph,
        nodelist=node_list,
        weight="combined_score",
        dtype=np.float32,
    )
    adjacency = 0.5 * (adjacency + adjacency.T) / 1000.0
    degree = adjacency.sum(axis=1)
    d_inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    return d_inv_sqrt[:, None] * adjacency * d_inv_sqrt[None, :]


def normalize_w_matrix(w_matrix):
    w_abs = np.abs(np.asarray(w_matrix, dtype=np.float32))
    threshold = np.percentile(w_abs, 50)
    w_abs[w_abs < threshold] = 0.0
    w_abs_max = float(np.abs(w_matrix).max())
    if w_abs_max <= 0:
        return w_abs
    return w_abs / w_abs_max


def safe_mean_positive(matrix):
    positive = matrix[matrix > 0]
    if positive.size == 0:
        return 1.0
    return float(np.mean(positive))


def summarize_spectral_modules(node_df):
    rows = []
    for module_id, group in node_df.groupby("Module"):
        mpg_group = group[group["node_type"] == "MPG"]
        mrm_group = group[group["node_type"] == "MRM"]
        rows.append(
            {
                "Module": module_id,
                "n_nodes": len(group),
                "n_mpg": len(mpg_group),
                "n_mrm": len(mrm_group),
                "mean_z_score": group["z_score"].mean(),
                "top_mpg": ";".join(
                    mpg_group.sort_values("z_score", ascending=False)["node_name"].head(10)
                ),
                "top_mrm": ";".join(
                    mrm_group.sort_values("z_score", ascending=False)["node_name"].head(10)
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["n_nodes", "mean_z_score"], ascending=[False, False])
        .reset_index(drop=True)
    )


def ensure_score_columns(df, score_fallback_cols=("z_score", "ipi_raw", "gene_score", "mrm_score")):
    df = df.copy()
    if "z_score" not in df.columns:
        for col in score_fallback_cols:
            if col in df.columns:
                df["z_score"] = pd.to_numeric(df[col], errors="coerce")
                break
        else:
            df["z_score"] = np.nan
    if "p_fdr" not in df.columns:
        df["p_fdr"] = np.nan
    if "is_significant" not in df.columns:
        df["is_significant"] = True
    return df


def annotate_modules(module_gene_df, gene_sets):
    try:
        import gseapy as gp
    except ImportError:
        gp = None

    all_enrich_results = []
    fixed_annotations = []
    candidate_by_module = {}
    for _, row in module_gene_df.iterrows():
        module_id = row["Module"]
        genes = row["mpg_genes"]
        if len(genes) == 0:
            fixed_annotations.append(
                {
                    "Module": module_id,
                    "module_annotation": "Too few MPG genes",
                    "annotation_term": np.nan,
                    "annotation_p_fdr": np.nan,
                    "annotation_overlap": np.nan,
                    "annotation_genes": "",
                }
            )
            continue

        if gp is None:
            fixed_annotations.append(
                {
                    "Module": module_id,
                    "module_annotation": f"Module {module_id}",
                    "annotation_term": np.nan,
                    "annotation_p_fdr": np.nan,
                    "annotation_overlap": np.nan,
                    "annotation_genes": "",
                }
            )
            continue

        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=gene_sets,
                organism="human",
                outdir=None,
            )
            res = enr.results.copy()
            if res.empty:
                raise ValueError("empty enrichment result")
            res["Module"] = module_id
            res["Adjusted P-value"] = pd.to_numeric(res["Adjusted P-value"], errors="coerce")
            res["Combined Score"] = pd.to_numeric(res.get("Combined Score", np.nan), errors="coerce")
            res["annotation_overlap_n"] = res.get("Overlap", "").map(overlap_hit_count)
            res = res.sort_values(["Adjusted P-value", "Combined Score"], ascending=[True, False])
            all_enrich_results.append(res)
            candidate_by_module[module_id] = res
        except Exception as exc:
            fixed_annotations.append(
                {
                    "Module": module_id,
                    "module_annotation": f"Module {module_id}",
                    "annotation_term": f"Enrichment failed: {type(exc).__name__}",
                    "annotation_p_fdr": np.nan,
                    "annotation_overlap": np.nan,
                    "annotation_genes": "",
                }
            )

    enrich_df = pd.concat(all_enrich_results, ignore_index=True) if all_enrich_results else pd.DataFrame()
    assigned_annotations = select_unique_annotations(candidate_by_module)
    annotation_df = pd.DataFrame(fixed_annotations + assigned_annotations)
    if not annotation_df.empty:
        annotation_df = annotation_df.sort_values("Module").reset_index(drop=True)
    return annotation_df, enrich_df


def select_unique_annotations(candidate_by_module):
    """Give each module the best pathway not already used in the same organ.

    Candidates are ordered by adjusted P, then Combined Score. When several
    modules propose the same pathway, it goes to the module with more
    overlapping genes, then the smaller adjusted P, then the higher Combined
    Score. A later pathway does not have to pass adjusted P < 0.05.
    """
    assigned_modules = set()
    assigned_terms = set()
    assigned_annotations = []
    if not candidate_by_module:
        return assigned_annotations

    while len(assigned_modules) < len(candidate_by_module):
        proposals = []
        for module_id, candidates in candidate_by_module.items():
            if module_id in assigned_modules:
                continue
            available = candidates[~candidates["Term"].isin(assigned_terms)]
            if not available.empty:
                proposals.append(available.iloc[0])
        if not proposals:
            break

        proposal_df = pd.DataFrame(proposals)
        for _, group in proposal_df.groupby("Term", sort=False):
            winner = group.sort_values(
                ["annotation_overlap_n", "Adjusted P-value", "Combined Score"],
                ascending=[False, True, False],
            ).iloc[0]
            assigned_annotations.append(annotation_record_from_row(winner, "unique_best_available"))
            assigned_modules.add(winner["Module"])
            assigned_terms.add(winner["Term"])

    for module_id, candidates in candidate_by_module.items():
        if module_id in assigned_modules:
            continue
        assigned_annotations.append(annotation_record_from_row(candidates.iloc[0], "duplicate_unavoidable"))
    return assigned_annotations


def annotation_record_from_row(candidate, assignment):
    return {
        "Module": candidate["Module"],
        "module_annotation": candidate["Term"],
        "annotation_term": candidate["Term"],
        "annotation_p_fdr": candidate["Adjusted P-value"],
        "annotation_overlap": candidate.get("Overlap", np.nan),
        "annotation_overlap_n": candidate.get("annotation_overlap_n", np.nan),
        "annotation_genes": candidate.get("Genes", ""),
        "annotation_assignment": assignment,
    }


def candidate_tables_from_enrichment(enrich_df):
    candidate_by_module = {}
    for module_id, res in enrich_df.groupby("Module"):
        res = res.copy()
        res["Adjusted P-value"] = pd.to_numeric(res["Adjusted P-value"], errors="coerce")
        res["Combined Score"] = pd.to_numeric(res.get("Combined Score", np.nan), errors="coerce")
        res["annotation_overlap_n"] = res.get("Overlap", "").map(overlap_hit_count)
        res = res.sort_values(["Adjusted P-value", "Combined Score"], ascending=[True, False])
        candidate_by_module[module_id] = res
    return candidate_by_module


def overlap_hit_count(value):
    if pd.isna(value):
        return 0
    text = str(value)
    match = re.match(r"(\d+)\s*/", text)
    if match:
        return int(match.group(1))
    genes = [gene for gene in text.split(";") if gene.strip()]
    return len(genes)


ANNOTATION_COLUMNS = [
    "Module",
    "module_annotation",
    "annotation_term",
    "annotation_p_fdr",
    "annotation_overlap",
    "annotation_genes",
    "annotation_overlap_n",
    "annotation_assignment",
]


def reassign_saved_annotations(organ_dir):
    spectral_dir = organ_dir / "spectral_modules"
    enrich_df = pd.read_csv(spectral_dir / "module_enrichment.tsv", sep="\t")
    old_annotations = pd.read_csv(spectral_dir / "module_annotations.tsv", sep="\t")
    assigned = pd.DataFrame(select_unique_annotations(candidate_tables_from_enrichment(enrich_df)))
    unchanged = old_annotations[~old_annotations["Module"].isin(assigned["Module"])].copy()
    annotation_df = pd.concat([unchanged, assigned], ignore_index=True)
    for column in ANNOTATION_COLUMNS:
        if column not in annotation_df.columns:
            annotation_df[column] = np.nan
    annotation_df = annotation_df[ANNOTATION_COLUMNS].sort_values("Module").reset_index(drop=True)
    old_by_module = old_annotations.set_index("Module")["module_annotation"]
    changes = []
    for row in assigned.itertuples(index=False):
        previous = old_by_module.get(row.Module)
        if str(previous) != str(row.module_annotation):
            changes.append((int(row.Module), previous, row.module_annotation, row.annotation_assignment))
    if not changes:
        return changes
    annotation_df.to_csv(spectral_dir / "module_annotations.tsv", sep="\t", index=False)

    plot_path = spectral_dir / "plot_df.tsv"
    if plot_path.exists() and not assigned.empty:
        plot_df = pd.read_csv(plot_path, sep="\t", dtype=str)
        module_key = "original_module_id" if "original_module_id" in plot_df.columns else "Module"
        new_by_original = {str(int(module)): name for module, name in zip(assigned["Module"], assigned["module_annotation"])}
        mapped = plot_df[module_key].map(new_by_original)
        plot_df["annotation"] = mapped.where(mapped.notna(), plot_df["annotation"])
        plot_df.to_csv(plot_path, sep="\t", index=False)
        annotation_by_refined = plot_df.groupby("module_id")["annotation"].first()
        for relative in (
            "refined_mrm/refined_mrm_list.tsv",
            "refined_mpg/refined_mpg_list.tsv",
            "refined_node_df/refined_node_df_list.tsv",
        ):
            path = organ_dir / relative
            if not path.exists():
                continue
            refined = pd.read_csv(path, sep="\t", dtype=str)
            refined["annotation"] = refined["module_id"].map(annotation_by_refined).where(
                refined["module_id"].map(annotation_by_refined).notna(),
                refined["annotation"],
            )
            refined.to_csv(path, sep="\t", index=False)
    return changes


def safe_name(value):
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return text or "module"


def main():
    parser = argparse.ArgumentParser(description="Run MPG/MRM spectral module analysis for one organ.")
    parser.add_argument("--prefix", default=None, help="HEST organ prefix, e.g. Breast")
    parser.add_argument(
        "--reassign-annotations",
        action="store_true",
        help="Reapply unique pathway assignment from saved enrichment tables without reclustering.",
    )
    parser.add_argument("--dataset", default="hest1k")
    parser.add_argument("--mpg_path", default=None)
    parser.add_argument("--mrm_path", default=None)
    parser.add_argument("--w_path", default=None)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--min_mpg_per_module", type=int, default=5)
    parser.add_argument("--min_mrm_per_module", type=int, default=5)
    parser.add_argument("--gene_sets", nargs="+", default=["MSigDB_Hallmark_2020"])
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Optional output directory. Defaults to 06-Module/<dataset>/<prefix>.",
    )
    args = parser.parse_args()

    if args.reassign_annotations:
        dataset_dir = PROJECT_ROOT / "06-Module" / args.dataset
        organs = [dataset_dir / args.prefix] if args.prefix else sorted(path for path in dataset_dir.iterdir() if path.is_dir())
        for organ_dir in organs:
            changes = reassign_saved_annotations(organ_dir)
            print(f"{organ_dir.name}: {len(changes)} assignment change(s)")
            for module_id, previous, current, assignment in changes:
                print(f"  M{module_id}: {previous} -> {current} ({assignment})")
        return
    if not args.prefix:
        parser.error("--prefix is required unless --reassign-annotations is set")

    prefix = args.prefix
    base_04 = PROJECT_ROOT / "04-MPG" / args.dataset / prefix
    base_05 = PROJECT_ROOT / "05-MRM" / args.dataset / prefix
    base_06 = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "06-Module" / args.dataset / prefix

    mpg_path = Path(args.mpg_path) if args.mpg_path else base_04 / "mpgs" / "mpgs_permutation_quantile_threshold_q1_gene_score.tsv"
    mrm_path = (
        Path(args.mrm_path)
        if args.mrm_path
        else base_05
        / "mrms"
        / "mrm_scores_from_mpg_weight-uniform_agg-abs_sum_permutation_empirical_pvalue_fdr_q0.05.tsv"
    )
    w_path = (
        Path(args.w_path)
        if args.w_path
        else PROJECT_ROOT / "03-MMC_network" / args.dataset / prefix / "W_matrix" / "learned_W_matrix.npy"
    )
    ppi_path = PROJECT_ROOT / "02-ppi_network" / "ppi_network" / f"{prefix}.pkl"
    mapping_path = HEST_ROOT / "processed_data" / "string_mapping" / f"string_mapping_{prefix}.tsv"
    a_mrm_path = PROJECT_ROOT / "01-morphology_network" / args.dataset / prefix / "A_M.npy"

    for required_path in [mpg_path, mrm_path, w_path, ppi_path, mapping_path, a_mrm_path]:
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    mpg_df = pd.read_csv(mpg_path, sep="\t")
    mrm_df = pd.read_csv(mrm_path, sep="\t")
    sig_mpg_df = mpg_df[mpg_df["is_significant"].astype(bool)].copy()
    sig_mrm_df = mrm_df[mrm_df["is_significant"].astype(bool)].copy()
    sig_mpg_df = ensure_score_columns(sig_mpg_df)
    sig_mrm_df = ensure_score_columns(sig_mrm_df)
    if sig_mpg_df.empty or sig_mrm_df.empty:
        raise ValueError(f"Need non-empty significant MPG/MRM tables for {prefix}.")

    genes_df = pd.read_csv(mapping_path, sep="\t", index_col=0)
    genes_df = genes_df.drop_duplicates(subset=["stringId"], keep="first").reset_index(drop=True)
    genes_df["gene_idx"] = np.arange(len(genes_df))
    gene_to_idx = dict(zip(genes_df["queryItem"], genes_df["gene_idx"]))
    gene_to_string = dict(zip(genes_df["queryItem"], genes_df["stringId"]))

    with open(ppi_path, "rb") as file:
        ppi_graph = pickle.load(file)
    a_mrm_full = np.load(a_mrm_path).astype(np.float32)
    w_full = np.load(w_path).astype(np.float32)

    sig_mpg_df = sig_mpg_df[sig_mpg_df["gene"].isin(gene_to_idx)].copy()
    sig_mpg_df["gene_idx"] = sig_mpg_df["gene"].map(gene_to_idx).astype(int)
    sig_mpg_df["string_id"] = sig_mpg_df["gene"].map(gene_to_string)
    sig_mrm_df["feature_idx"] = sig_mrm_df["feature_idx"].astype(int)
    sig_mrm_df = sig_mrm_df[sig_mrm_df["feature_idx"].between(0, w_full.shape[1] - 1)].copy()

    mpg_gene_indices = sig_mpg_df["gene_idx"].to_numpy(dtype=int)
    mpg_string_ids = sig_mpg_df["string_id"].tolist()
    mrm_feature_indices = sig_mrm_df["feature_idx"].to_numpy(dtype=int)
    if len(mpg_gene_indices) == 0 or len(mrm_feature_indices) == 0:
        raise ValueError(f"No valid MPG/MRM subset remains for {prefix}.")

    a_mpg_sub_norm = normalized_adjacency_from_nx(ppi_graph, mpg_string_ids)
    a_mrm_sub_norm = np.clip(a_mrm_full[np.ix_(mrm_feature_indices, mrm_feature_indices)], 0.0, 1.0)
    w_sub_norm = normalize_w_matrix(w_full[np.ix_(mpg_gene_indices, mrm_feature_indices)])
    w_mean = safe_mean_positive(w_sub_norm)
    scaler = (safe_mean_positive(a_mrm_sub_norm) + safe_mean_positive(a_mpg_sub_norm)) / w_mean / 2.0
    w_sub_norm = w_sub_norm * scaler

    s_matrix = np.block(
        [
            [a_mpg_sub_norm, w_sub_norm],
            [w_sub_norm.T, a_mrm_sub_norm],
        ]
    ).astype(np.float32)
    s_matrix = 0.5 * (s_matrix + s_matrix.T)
    s_matrix = np.nan_to_num(s_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    s_affinity = np.abs(s_matrix)
    np.fill_diagonal(s_affinity, 0.0)

    mpg_node_df = sig_mpg_df.copy()
    mpg_node_df["node_type"] = "MPG"
    mpg_node_df["node_name"] = mpg_node_df["gene"]
    mpg_node_df["local_idx"] = np.arange(len(mpg_node_df))
    mpg_node_df["global_idx"] = mpg_node_df["local_idx"]

    mrm_node_df = sig_mrm_df.copy()
    mrm_node_df["node_type"] = "MRM"
    mrm_node_df["node_name"] = "MRM_" + mrm_node_df["feature_idx"].astype(str)
    mrm_node_df["local_idx"] = np.arange(len(mrm_node_df))
    mrm_node_df["global_idx"] = len(mpg_node_df) + mrm_node_df["local_idx"]

    node_df = pd.concat(
        [
            mpg_node_df[["global_idx", "local_idx", "node_type", "node_name", "z_score", "p_fdr", "is_significant"]],
            mrm_node_df[["global_idx", "local_idx", "node_type", "node_name", "z_score", "p_fdr", "is_significant"]],
        ],
        ignore_index=True,
    )

    graph = nx.from_numpy_array(s_matrix)
    largest_component = sorted(nx.connected_components(graph), key=len, reverse=True)[0]
    largest_idx = np.array(sorted(largest_component))
    a_lcc = s_matrix[np.ix_(largest_idx, largest_idx)]
    a_lcc_affinity = s_affinity[np.ix_(largest_idx, largest_idx)]
    node_lcc_df = node_df.iloc[largest_idx].copy().reset_index(drop=True)

    n_clusters = min(args.n_clusters, len(node_lcc_df))
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="discretize",
        random_state=args.random_state,
    )
    node_lcc_df["Module"] = spectral.fit_predict(a_lcc_affinity)
    spectral_summary = summarize_spectral_modules(node_lcc_df)

    spectral_dir = base_06 / "spectral_modules"
    spectral_dir.mkdir(parents=True, exist_ok=True)
    node_lcc_df.to_csv(spectral_dir / "spectral_module_nodes.tsv", sep="\t", index=False)
    spectral_summary.to_csv(spectral_dir / "spectral_module_summary.tsv", sep="\t", index=False)

    module_gene_rows = []
    for module_id, group in node_lcc_df.groupby("Module"):
        mpg_group = group[group["node_type"] == "MPG"].copy()
        genes = mpg_group.sort_values("z_score", ascending=False)["node_name"].tolist()
        module_gene_rows.append(
            {
                "Module": module_id,
                "n_mpg": len(genes),
                "mpg_genes": genes,
                "top_mpg_genes": ";".join(genes[:10]),
            }
        )
    module_gene_df = pd.DataFrame(module_gene_rows)
    annotation_df, enrich_df = annotate_modules(module_gene_df, args.gene_sets)
    annotation_df.to_csv(spectral_dir / "module_annotations.tsv", sep="\t", index=False)
    if not enrich_df.empty:
        enrich_df.to_csv(spectral_dir / "module_enrichment.tsv", sep="\t", index=False)

    module_counts = (
        node_lcc_df.groupby("Module")
        .agg(
            n_nodes=("node_type", "size"),
            n_mpg=("node_type", lambda values: (values == "MPG").sum()),
            n_mrm=("node_type", lambda values: (values == "MRM").sum()),
        )
        .reset_index()
    )
    valid_modules = module_counts[
        (module_counts["n_mpg"] >= args.min_mpg_per_module)
        & (module_counts["n_mrm"] >= args.min_mrm_per_module)
    ].copy()
    valid_module_ids = set(valid_modules["Module"])
    module_id_map = {
        old_module_id: new_module_id
        for new_module_id, old_module_id in enumerate(sorted(valid_module_ids))
    }
    valid_modules["module_id"] = valid_modules["Module"].map(module_id_map).astype(int)
    annotation_keep = annotation_df[["Module", "module_annotation"]].copy()
    module_node_df = node_lcc_df.copy()

    def refine_node_table(node_type):
        refined = module_node_df[
            (module_node_df["node_type"] == node_type)
            & (module_node_df["Module"].isin(valid_module_ids))
        ].copy()
        refined = refined.merge(valid_modules[["Module", "module_id", "n_mpg", "n_mrm", "n_nodes"]], on="Module", how="left")
        refined = refined.merge(annotation_keep, on="Module", how="left")
        refined = refined.rename(columns={"Module": "original_module_id", "module_annotation": "annotation"})
        return refined[
            ["module_id", "annotation", "n_mpg", "n_mrm", "n_nodes", "node_name", "z_score", "p_fdr"]
        ].sort_values(["module_id", "z_score"], ascending=[True, False])

    refined_mrm_out = refine_node_table("MRM")
    refined_mpg_out = refine_node_table("MPG")
    refined_node_out = module_node_df[module_node_df["Module"].isin(valid_module_ids)].copy()
    refined_node_out = refined_node_out.merge(valid_modules[["Module", "module_id", "n_mpg", "n_mrm", "n_nodes"]], on="Module", how="left")
    refined_node_out = refined_node_out.merge(annotation_keep, on="Module", how="left")
    refined_node_out = refined_node_out.rename(columns={"Module": "original_module_id", "module_annotation": "annotation"})
    refined_node_out = refined_node_out[
        ["module_id", "annotation", "n_mpg", "n_mrm", "n_nodes", "node_name", "z_score", "p_fdr"]
    ].sort_values(["module_id", "z_score"], ascending=[True, False])

    refined_mrm_dir = base_06 / "refined_mrm"
    refined_mpg_dir = base_06 / "refined_mpg"
    refined_node_dir = base_06 / "refined_node_df"
    refined_mrm_dir.mkdir(parents=True, exist_ok=True)
    refined_mpg_dir.mkdir(parents=True, exist_ok=True)
    refined_node_dir.mkdir(parents=True, exist_ok=True)
    refined_mrm_out.to_csv(refined_mrm_dir / "refined_mrm_list.tsv", sep="\t", index=False)
    refined_mpg_out.to_csv(refined_mpg_dir / "refined_mpg_list.tsv", sep="\t", index=False)
    refined_node_out.to_csv(refined_node_dir / "refined_node_df_list.tsv", sep="\t", index=False)

    if not refined_node_out.empty:
        plot_node_df = module_node_df[module_node_df["Module"].isin(valid_module_ids)].copy()
        name_to_lcc_idx = {name: idx for idx, name in enumerate(node_lcc_df["node_name"])}
        plot_node_df["lcc_idx"] = plot_node_df["node_name"].map(name_to_lcc_idx)
        refined_idx = plot_node_df["lcc_idx"].astype(int).to_numpy()
        a_refined = a_lcc[np.ix_(refined_idx, refined_idx)].astype(np.float32)
        if len(plot_node_df) > 2:
            embedding = SpectralEmbedding(
                n_components=2,
                affinity="precomputed",
                random_state=args.random_state,
            )
            coords_refined = embedding.fit_transform(np.abs(a_refined))
            plot_node_df = plot_node_df.reset_index(drop=True)
            plot_node_df["SE1"] = coords_refined[:, 0]
            plot_node_df["SE2"] = coords_refined[:, 1]
            plot_node_df = plot_node_df.merge(
                refined_node_out[["node_name", "module_id", "annotation"]],
                on="node_name",
                how="left",
            )
            plot_node_df = plot_node_df.rename(columns={"Module": "original_module_id"})
            plot_node_df["Module"] = plot_node_df["module_id"]
            plot_node_df.to_csv(spectral_dir / "plot_df.tsv", sep="\t", index=False)

    print("-" * 30)
    print(f"Prefix: {prefix}")
    print(f"Significant MPG: {len(sig_mpg_df)}")
    print(f"Significant MRM: {len(sig_mrm_df)}")
    print(f"Refined MPG: {len(refined_mpg_out)}")
    print(f"Refined MRM: {len(refined_mrm_out)}")
    print(f"Modules kept: {refined_node_out['module_id'].nunique() if not refined_node_out.empty else 0}")
    print(f"Saved outputs under: {base_06}")
    print("-" * 30)


if __name__ == "__main__":
    main()
