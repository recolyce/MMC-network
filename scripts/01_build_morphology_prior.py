"""
Build the morphology prior network.

Read UNI features from pathology slides of one cancer, estimate dependence
among feature dimensions with Ledoit-Wolf, keep the strongest edges, partition
the dimensions with Leiden, and save the adjacency and Laplacian matrices.
"""

# One .pt file per slide: UNI patch features for this TCGA cancer.
UNI_FEATURE_GLOB = "<TCGA frozen UNI features for {disease}>/*.pt"
# Used when the directory above is empty: 20x UNI features for the same cancer.
UNI_FEATURE_FALLBACK_GLOB = "<TCGA 20x UNI features for {disease_lower}>/*.pt"
# Output directory for this organ's morphology prior: A_M, L_M, and module labels.
MORPHOLOGY_PRIOR_DIR = "<morphology prior output for {prefix}>"

import numpy as np
import igraph as ig
import leidenalg as la
from sklearn.covariance import GraphicalLasso, LedoitWolf
from sklearn.preprocessing import StandardScaler
import os

def build_morphology_prior_with_leiden(X, alpha=0.1):
    """
    X: PFM feature matrix, shape (N_samples, 1024).
    alpha: GraphicalLasso sparsity penalty. Larger values give fewer edges.
    """
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # Step 1: estimate the precision matrix.
    print(f"Estimating the precision matrix (dimension: {X.shape[1]})...")
    # try:
    #     gl = GraphicalLasso(alpha=alpha, max_iter=500, mode='cd')
    #     gl.fit(X)
    #     precision = gl.precision_
    # except Exception as e:
    #     print(f"GraphicalLasso failed: {e}. Falling back to Ledoit-Wolf.")
    # Ledoit-Wolf is the stable estimate used here.
    lw = LedoitWolf().fit(X)
    precision = lw.get_precision()

    adj_matrix = np.abs(precision)
    np.fill_diagonal(adj_matrix, 0)
    
    # Step 2: convert the matrix to an igraph object.
    # Keep the strongest edges to reduce noise.
    threshold = np.percentile(adj_matrix, 90)  # keep the top 10% of edges
    adj_matrix[adj_matrix < threshold] = 0
    
    # Indices of the remaining edges.
    sources, targets = adj_matrix.nonzero()
    weights = adj_matrix[sources, targets]
    
    # Build the graph.
    g = ig.Graph(n=adj_matrix.shape[0], edges=list(zip(sources, targets)), directed=False)
    g.es['weight'] = weights
    
    # Step 3: Leiden community detection.
    print("Running Leiden to partition morphology modules...")
    partition = la.find_partition(g, la.ModularityVertexPartition, weights='weight')
    
    module_labels = np.array(partition.membership)
    
    # Step 4: build the Laplacian L_M.
    # Strengthen edges that fall inside a Leiden module.
    A_M = adj_matrix.copy()
    for i in range(len(module_labels)):
        for j in range(len(module_labels)):
            if module_labels[i] == module_labels[j] and A_M[i,j] > 0:
                A_M[i,j] *= 1.5  # upweight within-module edges
    A_M = np.clip(A_M, 0, 1)
    D_M = np.diag(np.sum(A_M, axis=1))
    L_M = D_M - A_M
    # Step 5: symmetric normalization.
    # 1. Degree vector.
    row_sum = np.sum(A_M, axis=1)

    # 2. D^{-1/2}. Zero-degree nodes stay at zero.
    d_inv_sqrt = np.power(row_sum, -0.5, where=row_sum != 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    D_inv_sqrt = np.diag(d_inv_sqrt)

    # 3. A_norm = D^{-1/2} @ A @ D^{-1/2}
    A_M_norm = D_inv_sqrt @ A_M @ D_inv_sqrt

    # 4. Normalized Laplacian: L = I - A_norm
    L_M_norm = np.eye(A_M.shape[0]) - A_M_norm

    return L_M_norm, A_M_norm, A_M, L_M, module_labels

# Example run.
from glob import glob
from tqdm import tqdm
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Build the morphology prior matrices.")
parser.add_argument("--prefix", type=str, default="Pancreas", help="Organ prefix, for example Breast.")
parser.add_argument("--disease", type=str, default="PAAD", help="TCGA disease code, for example BRCA.")
args = parser.parse_args()
prefix = args.prefix
disease = args.disease
all_features = []
pt_files = glob(UNI_FEATURE_GLOB.format(disease=disease))
if len(pt_files) == 0:
    pt_files = glob(UNI_FEATURE_FALLBACK_GLOB.format(disease_lower=disease.lower()))
for pt_file in tqdm(pt_files):
    features = torch.load(pt_file)
    if features.shape[0] > 2000:
        indices = np.random.choice(features.shape[0], 2000, replace=False)
        features = features[indices]
    all_features.append(features)

features_matrix = torch.cat(all_features, dim=0).numpy()
print(f'feature matrix shape: {features_matrix.shape}')
L_M_norm, A_M_norm, A_M, L_M, labels = build_morphology_prior_with_leiden(features_matrix)

out_dir = MORPHOLOGY_PRIOR_DIR.format(prefix=prefix)
os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, "L_M_norm.npy"), L_M_norm)
np.save(os.path.join(out_dir, "A_M_norm.npy"), A_M_norm)
np.save(os.path.join(out_dir, "L_M.npy"), L_M)
np.save(os.path.join(out_dir, "A_M.npy"), A_M)
np.save(os.path.join(out_dir, "morph_labels.npy"), labels)