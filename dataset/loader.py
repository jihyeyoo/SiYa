import os
import h5py
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import gc


# -------------------------------------------------------
# Custom Sample class 
# -------------------------------------------------------

class CustomSample:

    def __init__(self, root, sample_id, metadata_dir=None):

        self.sample_id = sample_id
        self.st_path = os.path.join(root, "st", f"{sample_id}.h5ad")
        self.patch_path = os.path.join(root, "patches", f"{sample_id}.h5")

        if not os.path.exists(self.st_path):
            raise FileNotFoundError(f"{self.st_path} not found.")

        if not os.path.exists(self.patch_path):
            raise FileNotFoundError(f"{self.patch_path} not found.")

        # Load label from metadata JSON
        self.label = self._load_label_from_metadata(metadata_dir)

    def _load_label_from_metadata(self, metadata_dir):
        """Load label from metadata JSON"""

        if metadata_dir is None:
            metadata_dir = "./hest_data/metadata"

        meta_path = os.path.join(metadata_dir, f"{self.sample_id}.json")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        import json
        with open(meta_path) as f:
            meta = json.load(f)

        # Label mapping
        LABEL_MAP = {'Healthy': 0, 'Cancer': 1, 'Tumor': 1}

        label_str = meta.get("disease_state", -1)
        label_num = LABEL_MAP.get(label_str, -1)

        if label_num == -1:
            raise ValueError(f"Invalid label for {self.sample_id}: {label_str}")

        return label_num


# -------------------------------------------------------
# WSI-level Dataset
# -------------------------------------------------------

class WSIDataset(Dataset):

    def __init__(self, samples, max_spots=2000, hvg_genes=None):

        self.samples = samples
        self.max_spots = max_spots
        self.hvg_genes = hvg_genes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        try:

            # Load ST data
            adata = sc.read_h5ad(sample.st_path, backed='r')

            adata = adata.to_memory()

            # Fold-specific HVG slice
            if self.hvg_genes is not None:

                # 현재 sample gene
                sample_genes = list(adata.var_names)

                # gene → index
                gene_to_idx = {g: i for i, g in enumerate(sample_genes)}

                # expression matrix
                X_raw = adata.X

                if hasattr(X_raw, "toarray"):
                    expr_raw = X_raw.toarray()
                elif hasattr(X_raw, "todense"):
                    expr_raw = np.array(X_raw.todense())
                else:
                    expr_raw = np.asarray(X_raw)

                # output matrix (spots × global_hvg)
                expr = np.zeros((expr_raw.shape[0], len(self.hvg_genes)), dtype=np.float32)

                # fill existing genes
                for j, g in enumerate(self.hvg_genes):
                    if g in gene_to_idx:
                        expr[:, j] = expr_raw[:, gene_to_idx[g]]

                # overlap warning
                # overlap = sum(g in gene_to_idx for g in self.hvg_genes)
                # if overlap < len(self.hvg_genes) * 0.5:
                #     print(f"⚠️ {sample.sample_id}: HVG overlap {overlap}/{len(self.hvg_genes)}")

            else:

                expr = expr_raw

                X_raw = adata.X

                if hasattr(X_raw, "toarray"):
                    expr = X_raw.toarray()
                elif hasattr(X_raw, "todense"):
                    expr = np.array(X_raw.todense())
                else:
                    expr = np.asarray(X_raw)

            expr = expr.astype(np.float32)

            barcodes_st = adata.obs_names.to_numpy()
            coords = np.array(adata.obsm["spatial"])
            label_val = sample.label  # From metadata

            del adata

            # Load patches
            with h5py.File(sample.patch_path, "r") as f:
                imgs = f["img"][:]
                raw_bar = np.array(f["barcode"])

            patch_barcodes = [
                b.decode() if isinstance(b, (bytes, np.bytes_)) else str(b)
                for b in raw_bar.flatten()
            ]

            b2i = {b: i for i, b in enumerate(patch_barcodes)}

            patch_idx, st_idx = [], []

            for i, b in enumerate(barcodes_st):
                if b in b2i:
                    patch_idx.append(b2i[b])
                    st_idx.append(i)

            if len(patch_idx) == 0:
                raise ValueError("No aligned spots")

            images = imgs[patch_idx]
            expr = expr[st_idx]
            coords = coords[st_idx]

            # Spot sampling
            if self.max_spots and len(images) > self.max_spots:
                sel = np.random.choice(len(images), self.max_spots, replace=False)
                images, expr, coords = images[sel], expr[sel], coords[sel]

            # Tensor conversion
            images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.
            expr = torch.from_numpy(expr).float()
            coords = torch.from_numpy(coords).float()

            # Coordinate normalization
            if coords.shape[0] > 1:

                c_min = coords.min(dim=0, keepdim=True).values
                c_max = coords.max(dim=0, keepdim=True).values

                c_range = c_max - c_min
                c_range[c_range == 0] = 1.0

                coords = (coords - c_min) / c_range

            return {
                "images": images,
                "expr": expr,
                "coords": coords,
                "label": torch.tensor(label_val).long(),
                "sample_id": sample.sample_id,
                "num_spots": len(images)
            }

        finally:
            gc.collect()


# -------------------------------------------------------
# Dataloader creator
# -------------------------------------------------------

def create_wsi_dataloader(
    samples,
    batch_size=1,
    shuffle=True,
    max_spots=2000,
    hvg_genes=None
):

    dataset = WSIDataset(
        samples,
        max_spots=max_spots,
        hvg_genes=hvg_genes
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda x: x[0]
    )

    return loader
