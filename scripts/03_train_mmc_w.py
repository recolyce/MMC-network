"""
Learn the morphology-to-gene coupling matrix W.

Train a linear map from the morphology prior, the PPI prior, and this organ's
spatial transcriptomics profiles. Round 0 uses the true pairs. Each later round
permutes expression and trains a W for the null distribution.
"""

# HEST UNI features, one h5 file per sample.
HEST_UNI_H5_DIR = "<HEST UNI features, one h5 file per sample>"
# HEST expression matrices for this organ, one h5 file per sample.
HEST_COUNT_GLOB = "<HEST processed counts for {prefix}>/*.h5"
# Directory where the learned W and bias are written.
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
import copy
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import dual_graph_loss, load_al_from_nx


class CrossModalMapper(nn.Module):
    def __init__(self, dim_m, dim_g):
        super().__init__()
        self.W = nn.Linear(dim_m, dim_g)

    def forward(self, x_m):
        return self.W(x_m)


def get_permuted_dataset(original_dataset, rng):
    perm_dataset = copy.deepcopy(original_dataset)
    perm_idx = rng.permutation(len(perm_dataset.exprs))
    perm_dataset.exprs = perm_dataset.exprs[perm_idx]
    perm_dataset.masks = perm_dataset.masks[perm_idx]
    return perm_dataset


def get_subsampled_dataset(original_dataset, rng, max_samples):
    if max_samples is None or max_samples <= 0 or len(original_dataset.exprs) <= max_samples:
        return original_dataset

    subset_dataset = copy.deepcopy(original_dataset)
    subset_idx = rng.choice(len(subset_dataset.exprs), size=max_samples, replace=False)
    subset_dataset.morph_feats = subset_dataset.morph_feats[subset_idx]
    subset_dataset.exprs = subset_dataset.exprs[subset_idx]
    subset_dataset.masks = subset_dataset.masks[subset_idx]
    subset_dataset.names = [subset_dataset.names[i] for i in subset_idx]
    return subset_dataset


def train_one_round(
    dataset,
    dim_m,
    dim_g,
    l_morpho,
    l_molecule,
    device,
    batch_size,
    lr,
    weight_decay,
    alpha,
    beta,
    gamma,
    num_epochs,
):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = CrossModalMapper(dim_m, dim_g).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_reg_M = 0.0
        epoch_reg_G = 0.0
        epoch_l1_reg = 0.0

        for batch in dataloader:
            x_m = batch["morphology"].to(device, dtype=torch.float32)
            x_g = batch["molecule"].to(device, dtype=torch.float32)
            mask = batch["mask"].to(device, dtype=torch.bool)

            optimizer.zero_grad()
            x_g_pred = model(x_m)
            loss, mse, reg_M, reg_G, l1_reg = dual_graph_loss(
                x_g_pred,
                x_g,
                model.W.weight.T,
                l_morpho,
                l_molecule,
                mask,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += mse
            epoch_reg_M += reg_M
            epoch_reg_G += reg_G
            epoch_l1_reg += l1_reg

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == num_epochs:
            avg_loss = epoch_loss / len(dataloader)
            avg_mse = epoch_mse / len(dataloader)
            avg_reg_M = epoch_reg_M / len(dataloader)
            avg_reg_G = epoch_reg_G / len(dataloader)
            avg_l1_reg = epoch_l1_reg / len(dataloader)
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Total Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}"
                f"Reg_M: {avg_reg_M:.4f}, Reg_G: {avg_reg_G:.4f}, L1_reg: {avg_l1_reg:.4f}"
            )
    return model.W.weight.detach().cpu().numpy(), model.W.bias.detach().cpu().numpy()


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


def main():
    parser = argparse.ArgumentParser(
        description="Train MMC morphology-to-gene W/bias matrices for true and permutation rounds."
    )
    parser.add_argument("--prefix", type=str, default="Kidney", help="Dataset prefix")
    parser.add_argument("--dataset", type=str, default="hest1k")
    parser.add_argument("--n_perms", type=int, default=1000)
    parser.add_argument("--true_epochs", type=int, default=50)
    parser.add_argument("--perm_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument(
        "--perm_train_samples",
        type=int,
        default=8192,
        help="Number of randomly selected permuted samples used for training in each permutation round.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--nsamples",
        type=int,
        default=None,
        help="Use at most this many samples when building MMCData for quick runs.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip rounds whose W and bias files already exist.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    prefix = args.prefix
    dataset = args.dataset
    input_feat = HEST_UNI_H5_DIR
    input_omic_pattern = HEST_COUNT_GLOB.format(prefix=prefix)

    save_w_mat = W_DIR.format(dataset=dataset, prefix=prefix)
    os.makedirs(save_w_mat, exist_ok=True)

    a_morpho_np = np.load(MORPHOLOGY_ADJACENCY.format(prefix=prefix))
    a_morpho_np = np.clip(a_morpho_np, 0, 1)
    l_morpho_np = np.diag(np.sum(a_morpho_np, axis=1)) - a_morpho_np
    l_morpho = torch.tensor(l_morpho_np, dtype=torch.float32).to(device)

    genes_df = pd.read_csv(
        STRING_MAPPING.format(prefix=prefix),
        sep="\t",
        index_col=0,
    )
    genes_df.drop_duplicates(subset=["stringId"], keep="first", inplace=True)
    with open(PPI_GRAPH.format(prefix=prefix), "rb") as file:
        g_molecule = pickle.load(file)
    # _, l_molecule_np, _, _ = load_al_from_nx(g_molecule, genes_df["stringId"].tolist())
    _, _, _, l_molecule_np = load_al_from_nx(g_molecule, genes_df["stringId"].tolist())
    l_molecule = torch.tensor(l_molecule_np, dtype=torch.float32).to(device)

    mmc_dataset = MMCData(
        feat_dir=input_feat,
        omic_pth_pattern=input_omic_pattern,
        gene_list=genes_df["queryItem"].to_list(),
        preprocessing=True,
        dataset=dataset,
        num_sample=args.nsamples,
    )
    print(f"Length of training data: {len(mmc_dataset)}")

    dim_m = mmc_dataset.morph_feats.shape[1]
    dim_g = mmc_dataset.exprs.shape[1]

    for p in tqdm(range(0, args.n_perms + 1)):
        w_mat_out, w_bias_out = w_paths(save_w_mat, p)

        if args.skip_existing and os.path.exists(w_mat_out) and os.path.exists(w_bias_out):
            continue

        if p == 0:
            round_dataset = mmc_dataset
            num_epochs = args.true_epochs
            print("Round 0 / true data: ML training.")
        else:
            permuted_dataset = get_permuted_dataset(mmc_dataset, rng)
            round_dataset = get_subsampled_dataset(
                permuted_dataset,
                rng,
                args.perm_train_samples,
            )
            num_epochs = args.perm_epochs
            print(
                f"Permutation round {p} / {args.n_perms}: "
                f"ML training with {len(round_dataset)} sampled spots."
            )

        learned_w, learned_bias = train_one_round(
            dataset=round_dataset,
            dim_m=dim_m,
            dim_g=dim_g,
            l_morpho=l_morpho,
            l_molecule=l_molecule,
            device=device,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            num_epochs=num_epochs,
        )

        np.save(w_mat_out, learned_w)
        np.save(w_bias_out, learned_bias)


if __name__ == "__main__":
    main()
