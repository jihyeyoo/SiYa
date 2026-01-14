import os
import h5py
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# -------------------------------------------------------
# 0) Custom Sample class 
# -------------------------------------------------------
class CustomSample:
    def __init__(self, root, sample_id):
        """
        Load preprocessed sample
        - ST data is already normalized, log-transformed, HVG-filtered
        - Metadata is in adata.obs
        """
        self.sample_id = sample_id
        
        st_path = os.path.join(
            root,
            "st_preprocessed",
            f"{sample_id}.h5ad"
        )

        patch_path = os.path.join(
            root,
            "patches",
            f"{sample_id}.h5"
        )
        if not os.path.exists(st_path):
            raise FileNotFoundError(f"{st_path} not found.")
        if not os.path.exists(patch_path):
            raise FileNotFoundError(f"{patch_path} not found.")

        # Load preprocessed AnnData
        self.adata = sc.read_h5ad(st_path)
        # Already: normalized, log-transformed, HVG-filtered (2000 genes)
        
        # Patches
        self.patches = h5py.File(patch_path, "r")
        
        # Metadata from adata.obs (already embedded)
        # disease_state: 0 (Healthy) or 1 (Tumor)
        # No need for separate metadata.json!

# -------------------------------------------------------
# 1) WSI-level Dataset 
# -------------------------------------------------------
class WSIDataset(Dataset):
    def __init__(self, samples, max_spots=2000, use_label=True):
        """
        Args:
            samples: List of CustomSample objects
            max_spots: Max spots to load per WSI (memory saving)
            use_label: Whether to return labels
        """
        self.samples = samples
        self.max_spots = max_spots
        self.use_label = use_label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        adata = sample.adata

        # 1. ST data (already preprocessed!)
        # Already: normalized, log1p, HVG-filtered
        if hasattr(adata.X, 'toarray'):
            expr = adata.X.toarray()
        else:
            expr = adata.X
        
        barcodes_st = adata.obs_names.to_numpy()
        coords_st = adata.obsm["spatial"]

        # 2. Patch data
        patches = sample.patches
        imgs = patches["img"]
        raw_bar = np.array(patches["barcode"])
        
        patch_barcodes = [
            b.decode() if isinstance(b, bytes) else str(b)
            for b in raw_bar.squeeze()
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
            raise RuntimeError(f"No aligned spots in {sample.sample_id}")

        patch_indices = np.array(patch_indices)
        st_indices = np.array(st_indices)

        # Get aligned data
        images = imgs[patch_indices]
        expr = expr[st_indices]
        coords = coords_st[st_indices]
        
        # 4. Sampling (if too many spots)
        N = len(images)
        if self.max_spots is not None and N > self.max_spots:
            sample_indices = np.linspace(0, N-1, self.max_spots, dtype=int)
            images = images[sample_indices]
            expr = expr[sample_indices]
            coords = coords[sample_indices]

        # 5. Label (from adata.obs)
        if self.use_label:
            # disease_state already in adata.obs as 0 or 1
            # Take first value (all spots have same slide-level label)
            disease_state = adata.obs['disease_state'].values[0]
            
            if pd.isna(disease_state):
                label = torch.tensor(0).long()  # Default to Healthy
            else:
                label = torch.tensor(int(disease_state)).long()
        else:
            label = torch.tensor(-1).long()

        # 6. To Tensor
        images = torch.tensor(images).permute(0, 3, 1, 2).float() / 255.0
        expr = torch.tensor(expr).float()
        
        # Normalize coords to [0, 1]
        coords = torch.tensor(coords).float()
        if coords.shape[0] > 1:
            c_min = coords.min(dim=0, keepdim=True)[0]
            c_max = coords.max(dim=0, keepdim=True)[0]
            c_range = c_max - c_min
            c_range[c_range == 0] = 1.0
            coords = (coords - c_min) / c_range
        else:
            coords = torch.zeros_like(coords)

        return {
            "images": images,
            "expr": expr,
            "coords": coords,
            "label": label,
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
def create_wsi_dataloader(samples, batch_size=1, shuffle=True, max_spots=2000):
    dataset = WSIDataset(samples, max_spots=max_spots, use_label=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,  # RAM saving
        collate_fn=wsi_collate_fn
    )
    return loader

# -------------------------------------------------------
# 4) Get gene info 
# -------------------------------------------------------
def get_gene_info(samples):
    """
    Get gene information from preprocessed data
    All samples already have same genes (HVG-filtered to 2000)
    
    Returns:
        num_genes: Number of genes
        gene_names: List of gene names
    """
    sample = samples[0]
    num_genes = sample.adata.n_vars
    gene_names = sample.adata.var_names.tolist()
    
    print(f"Gene info:")
    print(f"  Total genes: {num_genes}")
    print(f"  Gene names (first 10): {gene_names[:10]}")
    
    return num_genes, gene_names

# -------------------------------------------------------
# 5) Main Check
# -------------------------------------------------------
if __name__ == "__main__":
    
    
    root_dir = "/workspace/Temp/ver1/hest_data"  # Adjust path
    
    # List all preprocessed samples
    st_dir = os.path.join(root_dir, "st_preprocessed")

    sample_ids = [
        f.replace(".h5ad", "")
        for f in os.listdir(st_dir)
        if f.endswith(".h5ad")
    ]

    
    print(f"Found {len(sample_ids)} preprocessed samples")
    print(f"Sample IDs: {sample_ids[:5]}...")
    
    # Load samples
    samples = []
    for sid in sample_ids[:5]:  # Test with first 5
        try:
            s = CustomSample(root_dir, sid)
            samples.append(s)
            print(f"✓ Loaded {sid}")
            print(f"  - Spots: {s.adata.n_obs}")
            print(f"  - Genes: {s.adata.n_vars}")
            print(f"  - Label: {s.adata.obs['disease_state'].values[0]}")
        except Exception as e:
            print(f"✗ Failed {sid}: {e}")

    if len(samples) > 0:
        # Get gene info
        num_genes, gene_names = get_gene_info(samples)
        
        # Create DataLoader
        loader = create_wsi_dataloader(samples, batch_size=1, max_spots=2000)
        
        # Check batch
        batch = next(iter(loader))
        print("\n" + "="*60)
        print("[Batch Structure Check]")
        print("="*60)
        print(f"Sample ID: {batch['sample_id']}")
        print(f"Images   : {batch['images'].shape}")
        print(f"Expr     : {batch['expr'].shape}")
        print(f"Coords   : {batch['coords'].shape}")
        print(f"Label    : {batch['label']}")
        print(f"Num spots: {batch['num_spots']}")
        print("="*60)
        print("✓ Sanity Check Passed!")
