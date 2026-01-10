import os
import json
import h5py
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.sparse as sp
import pandas as pd
from anndata import AnnData

# -------------------------------------------------------
# 0) Custom Sample class
# -------------------------------------------------------
class CustomSample:
    def __init__(self, root, sample_id):
        self.sample_id = sample_id
        
        st_path = os.path.join(root, sample_id, "st", "st.h5ad")
        patch_path = os.path.join(root, sample_id, "patches", "patches.h5")
        meta_path = os.path.join(root, sample_id, "metadata", "metadata.json")

        if not os.path.exists(st_path):
            raise FileNotFoundError(f"{st_path} not found.")
        if not os.path.exists(patch_path):
            raise FileNotFoundError(f"{patch_path} not found.")

        self.adata = sc.read_h5ad(st_path)
        
        # Make names unique for gene and spot
        self.adata.var_names_make_unique()
        self.adata.obs_names_make_unique()

        self.patches = h5py.File(patch_path, "r")

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

# -------------------------------------------------------
# 1) WSI-level Dataset
# -------------------------------------------------------
class WSIDataset(Dataset):
    def __init__(self, samples, gene_indices=None, use_label=True):
        """
        Args:
            samples (WSI from CustomSample)
            gene_indices: HVG index
            use_label
        """
        self.samples = samples
        self.gene_indices = gene_indices
        self.use_label = use_label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # one WSI
        sample = self.samples[idx]

        # 1. ST data
        adata = sample.adata
        
        # Expr matrix handling : spot_num * gene_num
        # Sparse to dense
        if sp.issparse(adata.X):
            expr = adata.X.toarray()
        else:
            expr = adata.X

        # Filter HVG gene with index
        if self.gene_indices is not None:
            expr = expr[:, self.gene_indices]

        barcodes_st = adata.obs_names.to_numpy()
        coords_st = adata.obsm["spatial"]

        # 2. Patch
        patches = sample.patches
        imgs = patches["img"]               # (N_patch, H, W, 3)
        raw_bar = patches["barcode"]
        
        # Decode barcodes
        raw_bar_np = np.array(raw_bar)
        patch_barcodes = [
            b.decode() if isinstance(b, bytes) else str(b)
            for b in raw_bar_np.squeeze()
        ]
        b2i = {b: i for i, b in enumerate(patch_barcodes)}

        # 3. Align ST ↔ Patch by barcode
        patch_indices = []
        st_indices = []

        for i, b in enumerate(barcodes_st):
            if b in b2i:
                patch_indices.append(b2i[b])
                st_indices.append(i)

        if len(patch_indices) == 0:
            # If no matching exists
            raise RuntimeError(f"No aligned spots found in {sample.sample_id}")

        patch_indices = np.array(patch_indices)
        st_indices = np.array(st_indices)

        # Get aligned data (ordered by ST)
        images = imgs[patch_indices]           # (N_matched, H, W, 3)
        expr = expr[st_indices]                # (N_matched, n_selected_genes)
        coords = coords_st[st_indices]         # (N_matched, 2)

        # 4. WSI-level label
        if self.use_label:
            meta = sample.metadata
            label_map = {"Healthy": 0, "Tumor": 1, "Cancer": 1}
            # If no disease_state key exists
            l_val = label_map.get(meta.get("disease_state", "Healthy"), 0)
            label = torch.tensor(l_val).long()
        else:
            label = torch.tensor(-1).long() # Dummy

        # 5. To Tensor & Normalize
        
        # Image: [0, 255] -> [0, 1], (H,W,C) -> (C,H,W)
        images = torch.tensor(images).permute(0, 3, 1, 2).float() / 255.0
        
        # Expr: Float tensor
        expr = torch.tensor(expr).float()
        
        # Coords: Normalize to [0, 1] per slide (Essential for spatial encoding)
        coords = torch.tensor(coords).float()
        if coords.shape[0] > 1:
            c_min = coords.min(dim=0, keepdim=True)[0]
            c_max = coords.max(dim=0, keepdim=True)[0]
            c_range = c_max - c_min
            c_range[c_range == 0] = 1.0 # division by zero 방지
            coords = (coords - c_min) / c_range
        else:
            coords = torch.zeros_like(coords)

        return {
            "images": images,       # (N_spots, 3, 224, 224)
            "expr": expr,           # (N_spots, n_hvg) -> scBERT input
            "coords": coords,       # (N_spots, 2) -> Spatial Token
            "label": label,         # (Scalar) -> WSI Classification Target
            "sample_id": sample.sample_id,
            "num_spots": len(images)
        }

# -------------------------------------------------------
# 2) Collate Fn
# -------------------------------------------------------
def wsi_collate_fn(batch):
    if len(batch) == 1:
        return batch[0]
    return batch

# -------------------------------------------------------
# 3) DataLoader Creator
# -------------------------------------------------------
def create_wsi_dataloader(samples, gene_indices=None, batch_size=1, shuffle=True):
    dataset = WSIDataset(samples, gene_indices=gene_indices, use_label=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0, 
        pin_memory=True,
        collate_fn=wsi_collate_fn
    )
    return loader

# -------------------------------------------------------
# 4) Preprocessing 
# -------------------------------------------------------
def preprocess_and_align_genes(samples):
    """
    Align common genes for each samples (WSI) after intersection 
    and normalize for scaling gene expression value. 
    """
    print("Alignment & Normalization...")
    
    # 1. Intersection
    common_genes = set(samples[0].adata.var_names)
    for s in samples[1:]:
        common_genes &= set(s.adata.var_names)
    
    common_genes = sorted(list(common_genes))
    print(f" - Common genes (Intersection): {len(common_genes)}")
    
    # If common genes are too small -> Unify
    if len(common_genes) < 100:
        print("Warning: Too few common genes!")
    
    # 2. Slice & Normalize each sample
    for s in samples:
        # Slice common genes
        s.adata = s.adata[:, common_genes].copy()
        
        # Normalization (Log1p)
        if np.max(s.adata.X) > 20: 
            sc.pp.normalize_total(s.adata, target_sum=1e4)
            sc.pp.log1p(s.adata)
            
    print(" - All samples aligned and normalized.")
    return common_genes

def select_hvg_indices(samples, n_top_genes=512):
    """
    Select HVG using scanpy 
    and return index for corresponding HVG
    """
    print(f"Selecting top {n_top_genes} HVGs...")
    
    # Concat
    adatas = [s.adata for s in samples]
    adata_concat = sc.concat(adatas, label="sample_id")
    
    # Compute HVG 
    sc.pp.highly_variable_genes(adata_concat, n_top_genes=n_top_genes, batch_key="sample_id")
    
    hvg_mask = adata_concat.var['highly_variable'].values
    hvg_indices = np.where(hvg_mask)[0] # Index for True spot
    hvg_names = adata_concat.var_names[hvg_mask].tolist()
    
    # Convert to tensor
    hvg_indices_t = torch.tensor(hvg_indices, dtype=torch.long)
    
    print(f" - Selected {len(hvg_indices_t)} genes.")
    return hvg_indices_t, hvg_names


# -------------------------------------------------------
# 5) Main 
# -------------------------------------------------------
if __name__ == "__main__":
    root_dir = "/workspace/Temp/ver1" 
    target_ids = ["TENX24", "TENX39"] 

    # 1. Load Samples
    samples = []
    for sid in target_ids:
        try:
            samples.append(CustomSample(root_dir, sid))
            print(f"Loaded {sid}")
        except Exception as e:
            print(f"Skipped {sid}: {e}")

    if len(samples) > 0:
        # 2. Preprocess (Align + Normalize)
        preprocess_and_align_genes(samples)

        # 3. Select HVG Indices
        hvg_idx, hvg_names = select_hvg_indices(samples, n_top_genes=512)

        # 4. Create DataLoader with HVG filter
        loader = create_wsi_dataloader(samples, gene_indices=hvg_idx, batch_size=1)

        # 5. Check Output
        batch = next(iter(loader))
        print("\n[Batch Structure Check]")
        print(f"Sample ID: {batch['sample_id']}")
        print(f"Images   : {batch['images'].shape}")  # (N, 3, 224, 224)
        print(f"Expr     : {batch['expr'].shape}")    # (N, 512) 
        print(f"Coords   : {batch['coords'].shape}")  # (N, 2)
        print(f"Label    : {batch['label']}")         # Scalar
        print("Sanity Check Passed!")
