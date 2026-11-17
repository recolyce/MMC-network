"""
Build the gene prior network.

Keep STRING human protein links whose proteins are in this organ's
identifier map, and save the graph used by MMC training.
"""

# STRING v12 human detailed protein-link table.
STRING_LINKS = "<STRING v12 human detailed protein links>"
# Gene-to-STRING identifier map for this organ.
STRING_MAPPING = "<STRING id mapping table for {prefix}>"
# Output pickle of this organ's PPI graph.
PPI_GRAPH = "<PPI graph pickle for {prefix}>"

import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description="Build a PPI network from STRING.")
parser.add_argument("--prefix", type=str, default="Pancreas", help="Organ prefix, for example Breast.")
args = parser.parse_args()
prefix = args.prefix

ppi_graph = PPI_GRAPH.format(prefix=prefix)
string_mapping = STRING_MAPPING.format(prefix=prefix)

info_df = pd.read_csv(STRING_LINKS, sep=' ', compression='gzip')
# Keep edges with coexpression score above 200, matching the saved PPI graph.
sub_info_df = info_df[info_df['coexpression'] > 200]

if os.path.exists(ppi_graph):
    print(f"{prefix}.pkl already exists; skipping network construction.")
else:
    genes_df = pd.read_csv(string_mapping, sep='\t', index_col=0)
    import pandas as pd
    import networkx as nx

    target_genes = set(genes_df['stringId'].dropna())

    filtered_ppi = sub_info_df[
        sub_info_df['protein1'].isin(target_genes) & 
        sub_info_df['protein2'].isin(target_genes)
    ]

    G = nx.from_pandas_edgelist(
        filtered_ppi, 
        source='protein1', 
        target='protein2', 
        edge_attr='combined_score'
    )

    G.add_nodes_from(target_genes)

    print("Network construction finished.")
    print(f"Nodes: {G.number_of_nodes()} (expected to equal the number of unique target genes)")
    print(f"Edges: {G.number_of_edges()}")

    sample_gene = list(target_genes)[0]
    if sample_gene in G:
        print(f"\nDegree of {sample_gene}: {G.degree(sample_gene)}")

    import pickle as pkl
    with open(ppi_graph, "wb") as f:
        pkl.dump(G, f)